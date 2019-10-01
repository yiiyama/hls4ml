//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_GARNET_H_
#define NNET_GARNET_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct garnet_config
{
  // Internal data type definitions
  typedef float bias_t;
  typedef float weight_t;
  typedef float accum_t;

  // Layer specs
  static const unsigned n_vertices = 250;
  static const unsigned n_in_features = 4;
  static const unsigned n_aggregators = 4;
  static const unsigned n_filters = 4;
  static const unsigned n_propagate = 4;

  // Resource reuse info
  static const unsigned io_type = io_parallel;
  static const unsigned reuse_factor = 1;
  static const bool store_weights_in_bram = false;
  static const unsigned n_zeros = 0;

  struct input_transform_config : public dense_config {
    static const unsigned n_in = 4; // n_in_features
    static const unsigned n_out = 4; // n_propagate
  };

  struct aggregator_distance_config : public dense_config {
    static const unsigned n_in = 4; // n_in_features
    static const unsigned n_out = 4; // n_aggregators
  };

  struct output_transform_config : public dense_config {
    static const unsigned n_in = 72; // 2 * n_aggregators * (n_propagate + n_aggregators) + n_in_features + n_aggregators
    static const unsigned n_out = 4; // n_filters
  };
};

template<class data_T, class res_T, typename CONFIG_T>
void garnet(
    data_T    data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    res_T     res[CONFIG_T::n_vertices * CONFIG_T::n_filters],
    typename CONFIG_T::weight_t  input_transform_weights[CONFIG_T::n_vertices * CONFIG_T::n_in_features * CONFIG_T::n_propagate],
    typename CONFIG_T::bias_t    input_transform_biases[CONFIG_T::n_vertices * CONFIG_T::n_propagate],
    typename CONFIG_T::weight_t  aggregator_distance_weights[CONFIG_T::n_vertices * CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::bias_t    aggregator_distance_biases[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
    typename CONFIG_T::weight_t  output_transform_weights[CONFIG_T::n_vertices * (CONFIG_T::n_in_features + 2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators+CONFIG_T::n_propagate) + CONFIG_T::n_aggregators) * CONFIG_T:: n_filters],
    typename CONFIG_T::bias_t    output_transform_biases[CONFIG_T::n_vertices * CONFIG_T::n_filters])
{
  // just to make the code a bit more readable - can replace all later if we need to
  unsigned const nvtx = CONFIG_T::n_vertices;
  unsigned const nfeat = CONFIG_T::n_in_features;
  unsigned const nprop = CONFIG_T::n_propagate;
  unsigned const naggr = CONFIG_T::n_aggregators;
  unsigned const nfilt = CONFIG_T::n_filters;
  unsigned const nlatent = nprop + naggr;

  // compute features and edge weights per vertex and save onto the aggregator array
  // should the aggregator get all nlatent values or just nprop? - Abhijay is checking
  // should we do mean and max (factor 2)? - Abhijay checking
  typename CONFIG_T::accum_t aggregated[2 * naggr * nlatent];

  for (int ia=0; ia<naggr; ia++){
    for (int il=0; il<nlatent; il++){
      aggregated[ia * nlatent + il] = 0.;
      aggregated[(naggr + ia) * nlatent + il] = 0.; // no need if we use just mean
    }
  }

  typename CONFIG_T::accum_t edge_weights[nvtx * naggr];

  for(int iv=0; iv<nvtx; iv++){
    typename CONFIG_T::accum_t features[nprop];
    typename CONFIG_T::accum_t* vertex_edge_weights = edge_weights + iv * naggr;
    
    dense_latency<data_T, res_T, typename CONFIG_T::input_transform_config>(
        data + iv * nfeat,
        features,
        input_transform_weights,
        input_transform_biases);

    dense_latency<data_T, res_T, typename CONFIG_T::aggregator_distance_config>(
        data + iv * nfeat,
        vertex_edge_weights,
        aggregator_edge_weights_weights,
        aggregator_edge_weights_biases);

    for (int ia=0; ia<naggr; ia++){
      vertex_edge_weights[ia] = pow(2, -vertex_edge_weights[ia]);
    }

    // aggregate

    for (int ia=0; ia<naggr; ia++){
      for (int ip=0; ip<nprop; ip++){
        // mean
        aggregated[ia * nlatent + ip] += features[ip] * vertex_edge_weights[ia];
        // max
        aggregated[(naggr + ia) * nlatent + ip] = max(aggregated[(naggr + ia) * nlatent + ip], features[ip] * vertex_edge_weights[ia]);
      }
      // see above - do we really need to concatenate vertex_edge_weights to features?
      for (int iw=0; iw<naggr; iw++){
        // mean
        aggregated[ia * nlatent + nprop + iw] += vertex_edge_weights[iw] * vertex_edge_weights[ia];
        // max
        aggregated[(naggr + ia) * nlatent + nprop + iw] = max(aggregated[(naggr + ia) * nlatent + nprop + iw], vertex_edge_weights[iw] * vertex_edge_weights[ia]);
      }
    }
  }

  for (int ia=0; ia<naggr; ia++){
    for (int il=0; il<nlatent; il++){
      aggregated[ia * nlatent + il] /= nvtx;
    }
  }

  for(int iv=0; iv<nvtx; iv++){
    // do we really need to concatenate all this?
    typename CONFIG_T::accum_t updated_features[2 * naggr * nlatent + nfeat + naggr];
    typename CONFIG_T::accum_t* vertex_edge_weights = edge_weights + iv * naggr;

    // return to vertices
    for (int ia=0; ia<naggr; ia++){
      for (int ip=0; ip<nprop; ip++){
        updated_features[ia * nlatent + ip] = aggregated[ia * nlatent + ip] * vertex_edge_weights[ia];
      }
      // see above - do we really need to concatenate edge_weights to features?
      for (int iw=0; iw<naggr; iw++){
        updated_features[(naggr + ia) * nlatent + nprop + iw] = aggregated[(naggr + ia) * nlatent + nprop + iw] * vertex_edge_weights[ia];
      }
    }

    // additional stuff to concatenate
    for (int ii=0; ii<nfeat; ii++){
      updated_features[2 * naggr * nlatent + ii] = data[ii];
    }
    for (int ia=0; ia<naggr; ia++){
      updated_features[2 * naggr * nlatent + nfeat + ia] = vertex_edge_weights[ia];
    }
    
    dense_latency<data_T, res_T, typename CONFIG_T::output_transform_config>(
        updated_features,
        res + iv * nfilt,
        output_transform_weights,
        output_transform_biases);
  }
}

}

#endif
