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

#define GARNET_COLLAPSE 1

#include "nnet_common.h"
#include "hls_stream.h"
#include "hls_math.h"

namespace nnet {

template<class CONFIG_T>
inline typename CONFIG_T::edge_weight_t
compute_garnet_edge_weight(typename CONFIG_T::accum_t distance)
{
  typename CONFIG_T::edge_weight_t edge_weight = 1.;
  return edge_weight >> static_cast<int>(distance);
}

struct garnet_config
{
  // Internal data type definitions
  typedef float input_transform_weights_t;
  typedef float input_transform_biases_t;
  typedef float output_transform_weights_t;
  typedef float output_transform_biases_t;
  typedef float aggregator_distance_weights_t;
  typedef float aggregator_distance_biases_t;

  typedef float accum_t;
  typedef ap_fixed<64, 60> edge_weight_t;

  typedef unsigned short index_t;

  // Layer specs
  static const unsigned n_vertices = 250;
  static const unsigned n_in_features = 4;
  static const unsigned n_aggregators = 4;
  static const unsigned n_filters = 4;
  static const unsigned n_propagate = 4;
};

template<class data_T, class res_T, typename CONFIG_T>
void garnet(
    data_T data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    data_T nvtx[1],
#ifdef GARNET_COLLAPSE
    res_T res[CONFIG_T::n_filters],
#else
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_filters],
#endif
    typename CONFIG_T::input_transform_weights_t input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_propagate],
    typename CONFIG_T::input_transform_biases_t input_transform_biases[CONFIG_T::n_propagate],
    typename CONFIG_T::aggregator_distance_weights_t aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_weights_t output_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_propagate * CONFIG_T::n_filters],
    typename CONFIG_T::output_transform_biases_t output_transform_biases[CONFIG_T::n_filters]
)
{
  typename CONFIG_T::index_t const n_latent = CONFIG_T::n_aggregators * CONFIG_T::n_propagate;
  typename CONFIG_T::accum_t const vnorm = 1. / CONFIG_T::n_vertices;

#ifdef GARNET_COLLAPSE
  typename CONFIG_T::accum_t edge_weight_sums[CONFIG_T::n_aggregators];
#else
  typename CONFIG_T::accum_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];
#endif
  typename CONFIG_T::accum_t aggregation_sum[n_latent];
  
#ifdef GARNET_COLLAPSE
 EdgeWeightSumInit:
  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    #pragma HLS UNROLL
    edge_weight_sums[ia] = 0.;
  }
#endif

 Vertices:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    if (iv < nvtx[0]) {
      typename CONFIG_T::index_t vertex_offset = iv * CONFIG_T::n_in_features;
  
      #pragma HLS EXPRESSION_BALANCE
      typename CONFIG_T::accum_t aggregator_distances[CONFIG_T::n_aggregators];
      typename CONFIG_T::accum_t aggregated_features[CONFIG_T::n_propagate];
  
      #pragma HLS PIPELINE
  
    DistInit:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        //#pragma HLS UNROLL
        aggregator_distances[ia] = aggregator_distance_biases[ia];
      }
    
    AggrInit:
      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        //#pragma HLS UNROLL
        aggregated_features[ip] = input_transform_biases[ip];
      }
  
    InputFeatures:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        //#pragma HLS PIPELINE
  
        typename CONFIG_T::index_t data_index = vertex_offset + ix;
  
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          //#pragma HLS UNROLL
          // keras Dense applies weights as K.dot(inputs, kernel) -> kernel is channels first
          typename CONFIG_T::index_t weight_index = ix * CONFIG_T::n_aggregators + ia;
          aggregator_distances[ia] += data[data_index] * aggregator_distance_weights[weight_index];
        }
  
        for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
          //#pragma HLS UNROLL
          typename CONFIG_T::index_t weight_index = ix * CONFIG_T::n_propagate + ip;
          aggregated_features[ip] += data[data_index] * input_transform_weights[weight_index];
        }
      }
  
    EdgeWeights:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        //#pragma HLS UNROLL

        typename CONFIG_T::edge_weight_t edge_weight = compute_garnet_edge_weight<CONFIG_T>(aggregator_distances[ia]);

#ifdef GARNET_COLLAPSE
        edge_weight_sums[ia] += edge_weight;
#else
        typename CONFIG_T::index_t index = iv * CONFIG_T::n_aggregators + ia;
        edge_weights[index] = edge_weight;
#endif
        for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
          //#pragma HLS UNROLL
          typename CONFIG_T::index_t il = ia * CONFIG_T::n_propagate + ip;
          aggregation_sum[il] += edge_weight * aggregated_features[ip];
        }
      }
    }
  }

#ifdef GARNET_COLLAPSE
  for (int io = 0; io < CONFIG_T::n_filters; ++io) {
    typename CONFIG_T::accum_t acc = 0.;

    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      #pragma HLS UNROLL
      typename CONFIG_T::edge_weight_t edge_weight_sum = edge_weight_sums[ia];
  
      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        #pragma HLS UNROLL
        typename CONFIG_T::index_t il = ia * CONFIG_T::n_propagate + ip;
        typename CONFIG_T::index_t weight_index = il * CONFIG_T::n_filters + io;
          
        acc += edge_weight_sum * aggregation_sum[il] * output_transform_weights[weight_index];
      }
    }

    #if GARNET_COLLAPSE == 1
    // mean
    acc *= vnorm * vnorm;
    acc += output_transform_biases[io] * nvtx[0] * vnorm;
    #elif GARNET_COLLAPSE == 2
    // sum
    acc += output_transform_biases[io] * nvtx[0];
    #endif

    res[io] = acc;
  }
#else
  for (unsigned il = 0; il < n_latent; ++il) {
    #pragma HLS UNROLL
    aggregation_sum[il] *= vnorm;
  }

  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    if (iv < nvtx[0]) {
      typename CONFIG_T::index_t vertex_offset = iv * CONFIG_T::n_filters;
      typename CONFIG_T::index_t edge_weight_offset = iv * CONFIG_T::n_aggregators;

      for (unsigned io = 0; io < CONFIG_T::n_filters; ++io) {
        #pragma HLS UNROLL

        typename CONFIG_T::accum_t acc = output_transform_biases[io];

        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          #pragma HLS UNROLL
  
          typename CONFIG_T::edge_weight_t edge_weight = edge_weights[edge_weight_offset + ia];
  
          for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
            #pragma HLS UNROLL
            typename CONFIG_T::index_t il = ia * CONFIG_T::n_propagate + ip;
            typename CONFIG_T::index_t weight_index = il * CONFIG_T::n_filters + io;
          
            acc += edge_weight * aggregation_sum[il] * output_transform_weights[weight_index];
          }
        }

        res[vertex_offset + io] = acc;
      }
    }
  }
#endif
}

}

#endif
