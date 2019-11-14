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
//#include "nnet_dense.h"
#include "hls_stream.h"
#include "hls_math.h"
#include <cmath>
//#include <math.h>
namespace nnet {

struct garnet_config
{
  // Internal data type definitions
  //typedef float bias_t;
  typedef float input_transform_weights_t;
  typedef float input_transform_biases_t;
  typedef float output_transform_weights_t;
  typedef float output_tranform_biases_t;
  typedef float aggregator_distance_weights_t;
  typedef float aggregator_distance_biases_t;

  typedef float accum_t;

  // Layer specs
  static const unsigned n_vertices = 250;
  static const unsigned n_in_features = 4;
  static const unsigned n_aggregators = 4;
  static const unsigned n_filters = 4;
  static const unsigned n_propagate = 4;

  // Resource reuse info
  //static const unsigned io_type = io_parallel;
  static const unsigned reuse_factor = 1;
  static const bool store_weights_in_bram = false;
  static const unsigned n_zeros = 0;

  struct input_transform_config : dense_config {
    static const unsigned n_in = n_in_features;
    static const unsigned n_out = n_propagate;
   // static const unsigned io_type = garnet_config::io_type;
    static const unsigned reuse_factor = garnet_config::reuse_factor;
    static const bool store_weights_in_bram = garnet_config::store_weights_in_bram;
  };

  struct aggregator_distance_config : dense_config {
    static const unsigned n_in = n_in_features;
    static const unsigned n_out = n_aggregators;
   // static const unsigned io_type = garnet_config::io_type;
    static const unsigned reuse_factor = garnet_config::reuse_factor;
    static const bool store_weights_in_bram = garnet_config::store_weights_in_bram;
  };

  struct output_transform_config : dense_config {
    //static const unsigned n_in =  n_aggregators * (n_propagate) + n_aggregators;
    static const unsigned n_in =  n_aggregators * (n_propagate);
    static const unsigned n_out = n_filters;
   // static const unsigned io_type = garnet_config::io_type;
    static const unsigned reuse_factor = garnet_config::reuse_factor;
    static const bool store_weights_in_bram = garnet_config::store_weights_in_bram;
  };
};

template<class data_T, class res_T, typename CONFIG_T>
void garnet(
    data_T    data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    res_T     res[CONFIG_T::n_vertices * CONFIG_T::n_filters],
    typename CONFIG_T::input_transform_weights_t  input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_propagate],
    typename CONFIG_T::input_transform_biases_t    input_transform_biases[CONFIG_T::n_propagate],
    typename CONFIG_T::aggregator_distance_weights_t  aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t    aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_weights_t  output_transform_weights[( CONFIG_T::n_aggregators * CONFIG_T::n_propagate) * CONFIG_T:: n_filters],
    typename CONFIG_T::output_transform_biases_t    output_transform_biases[CONFIG_T::n_filters])
{
  // just to make the code a bit more readable - can replace all later if we need to
  unsigned const nvtx = CONFIG_T::n_vertices;
  unsigned const nfeat = CONFIG_T::n_in_features;
  unsigned const nprop = CONFIG_T::n_propagate;
  unsigned const naggr = CONFIG_T::n_aggregators;
  unsigned const nfilt = CONFIG_T::n_filters;
  unsigned const nlatent = nprop;

  // compute features and edge weights per vertex and save onto the aggregator array
  // should the aggregator get all nlatent values or just nprop? - Abhijay is checking
  // should we do mean and max (factor 2)? - Abhijay checking
  typename CONFIG_T::accum_t aggregated[naggr * nlatent];

  for (int ia=0; ia<naggr; ia++){
    for (int il=0; il<nlatent; il++){
      aggregated[ia * nlatent + il] = 0.;
      //aggregated[(naggr + ia) * nlatent + il] = 0.; // no need if we use just mean
    }
  }

  typename CONFIG_T::accum_t edge_weights[nvtx * naggr];
  typename CONFIG_T::accum_t cache;
  typename CONFIG_T::accum_t features[nprop];

  Vertices1: for(int iv=0; iv<nvtx; iv++){
    typename CONFIG_T::accum_t* vertex_edge_weights = edge_weights + iv * naggr;

    // dense_latency<data_T, res_T, typename CONFIG_T::input_transform_config>(
    //     data + iv * nfeat,
    //     features,
    //     input_transform_weights,
    //     input_transform_biases);

    InputTransformOuter: for (int iout = 0; iout < nprop; iout++) {
      cache = input_transform_biases[iout];
      InputTransformInner: for (int iin = 0; iin < nfeat; iin++) {
        cache += input_transform_weights[iout * nfeat + iin] * data[iv * nfeat + iin];
      }
      features[iout] = cache;
    }
    
    // dense_latency<data_T, res_T, typename CONFIG_T::aggregator_distance_config>(
    //     data + iv * nfeat,
    //     vertex_edge_weights,
    //     aggregator_distance_weights,
    //     aggregator_distance_biases);

    AggregatorDistanceOuter: for (int iout = 0; iout < naggr; iout++) {
      cache = aggregator_distance_biases[iout];
      AggregatorDistanceInner: for (int iin = 0; iin < nfeat; iin++) {
        cache += aggregator_distance_weights[iout * nfeat + iin] * data[iv * nfeat + iin];
      }
      vertex_edge_weights[iout] = std::exp2f(-cache);
    }

    // for (int ia=0; ia<naggr; ia++){
    //   vertex_edge_weights[ia] = std::pow(2., -vertex_edge_weights[ia]);
    // }


    // aggregate

    AggregatorAccumOut: for (int ia=0; ia<naggr; ia++){
      AggregatorAccumIn: for (int ip=0; ip<nprop; ip++){
        // mean
        aggregated[ia * nlatent + ip] += features[ip] * vertex_edge_weights[ia];
        // max
	//if(aggregated[(naggr + ia) * nlatent + ip] < (features[ip] * vertex_edge_weights[ia])){
//		aggregated[(naggr + ia) * nlatent + ip] = features[ip] * vertex_edge_weights[ia]; 
//	}
        //aggregated[(naggr + ia) * nlatent + ip] = max(aggregated[(naggr + ia) * nlatent + ip], features[ip] * vertex_edge_weights[ia]);
      }
      // see above - do we really need to concatenate vertex_edge_weights to features?
     // for (int iw=0; iw<naggr; iw++){
        // mean
       // aggregated[ia * nlatent + nprop + iw] += vertex_edge_weights[iw] * vertex_edge_weights[ia];
        // max
//	if(aggregated[(naggr + ia) * nlatent + nprop + iw] < (vertex_edge_weights[iw] * vertex_edge_weights[ia])){
	
  //      aggregated[(naggr + ia) * nlatent + nprop + iw] =  vertex_edge_weights[iw] * vertex_edge_weights[ia];
//	}
     // }
    }
  }

  AggregatorMeanOut: for (int ia=0; ia<naggr; ia++){
    AggregatorMeanLatent: for (int il=0; il<nlatent; il++){
      aggregated[ia * nlatent + il] /= nvtx;
    }
  }

  // typename CONFIG_T::accum_t updated_features[naggr * nlatent +  naggr];
  typename CONFIG_T::accum_t updated_features[naggr * nlatent];

  Vertices2: for(int iv=0; iv<nvtx; iv++){
    // do we really need to concatenate all this?
    typename CONFIG_T::accum_t* vertex_edge_weights = edge_weights + iv * naggr;

    // return to vertices
    FeatureReturnAggr: for (int ia=0; ia<naggr; ia++){
      FeatureReturnProp: for (int ip=0; ip<nlatent; ip++){
        updated_features[ia * nlatent + ip] = aggregated[ia * nlatent + ip] * vertex_edge_weights[ia];
      }
    }

    // // additional stuff to concatenate
    // for (int ia=0; ia<naggr; ia++){
    //   updated_features[ naggr * nlatent + ia] = vertex_edge_weights[ia];
    // }
    
    // dense_latency<data_T, res_T, typename CONFIG_T::output_transform_config>(
    //     updated_features,
    //     res + iv * nfilt,
    //     output_transform_weights,
    //     output_transform_biases);

    OutputTransformOuter: for (int iout = 0; iout < nfilt; iout++) {
      cache = output_transform_biases[iout];
      OutputTransformInner: for (int iin = 0; iin < naggr * nlatent; iin++) {
        cache += output_transform_weights[iout * (naggr * nlatent) + iin] * updated_features[iin];
      }
      res[iv * nfilt + iout] = cache;
    }

  }
}

}

#endif
