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
#include "hls_stream.h"
#include "hls_math.h"

namespace nnet {

template<class CONFIG_T>
inline void
edge_weight_sums_init(typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators])
{
  #pragma HLS PIPELINE
 EdgeWeightSumInit:
  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    edge_weight_sums[ia] = 0.;
  }
}

template<class CONFIG_T>
inline void
aggregation_sums_init(typename CONFIG_T::aggr_t aggregation_sums[CONFIG_T::n_aggregators * CONFIG_T::n_propagate])
{
  #pragma HLS PIPELINE
 AggregationInit:
  for (unsigned il = 0; il < CONFIG_T::n_aggregators * CONFIG_T::n_propagate; ++il) {
    aggregation_sums[il] = 0.;
  }
}

template<class CONFIG_T>
inline typename std::enable_if<std::is_class<typename CONFIG_T::accum_t>::value>::type
initialize_edge_weights_table(typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  typedef ap_uint<CONFIG_T::distance_bitwidth> index_t;
  typedef ap_fixed<CONFIG_T::distance_bitwidth, CONFIG_T::distance_bitwidth / 2, AP_RND, AP_SAT> rdistance_t;

  unsigned const table_size = (1 << CONFIG_T::distance_bitwidth);

  index_t index;
  rdistance_t rdist;
  typename CONFIG_T::accum_t distance;
  
  for (unsigned iw = 0; iw < table_size; ++iw) {
    index = iw;
    rdist.range(CONFIG_T::distance_bitwidth - 1, 0) = index.range(CONFIG_T::distance_bitwidth - 1, 0);
    distance = rdist;
    edge_weights_table[iw] = hls::pow(2., -distance);
  }
}

template<class CONFIG_T>
inline typename std::enable_if<not std::is_class<typename CONFIG_T::accum_t>::value>::type
initialize_edge_weights_table(typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  unsigned const table_size = (1 << CONFIG_T::distance_bitwidth);
  double const step = 64. / table_size;

  typename CONFIG_T::accum_t v = -32.;
  for (unsigned iw = 0; iw < table_size; ++iw) {
    edge_weights_table[iw] = std::pow(2., -v);
    v += step;
  }
}

template<class CONFIG_T>
inline typename std::enable_if<std::is_class<typename CONFIG_T::accum_t>::value, typename CONFIG_T::edge_weight_t>::type
get_edge_weight(typename CONFIG_T::accum_t distance, typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  typedef ap_uint<CONFIG_T::distance_bitwidth> index_t;
  typedef ap_fixed<CONFIG_T::distance_bitwidth, CONFIG_T::distance_bitwidth / 2, AP_RND, AP_SAT> rdistance_t;

  index_t index;
  rdistance_t rdist = distance;

  index.range(CONFIG_T::distance_bitwidth - 1, 0) = rdist.range(CONFIG_T::distance_bitwidth - 1, 0);

  return edge_weights_table[index];
}

template<class CONFIG_T>
inline typename std::enable_if<not std::is_class<typename CONFIG_T::accum_t>::value, typename CONFIG_T::edge_weight_t>::type
get_edge_weight(typename CONFIG_T::accum_t distance, typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  unsigned const table_size = (1 << CONFIG_T::distance_bitwidth);
  double const step = 64. / table_size;
  
  int index = (distance + 32.) / step;
  if (index < 0)
    index = 0;
  else if (index >= table_size)
    index = table_size - 1;

  return edge_weights_table[index];
}

template<class CONFIG_T>
inline typename CONFIG_T::edge_weight_t
compute_garnet_edge_weight(typename CONFIG_T::accum_t distance)
{
  #pragma HLS PIPELINE
  // typename CONFIG_T::edge_weight_t edge_weight = 1.;
  // return edge_weight >> distance.to_int();

#ifdef __SYNTHESIS__
  typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_bitwidth];
  unsigned const reshape_factor = CONFIG_T::n_aggregators * CONFIG_T::n_in_features * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
  #pragma HLS ARRAY_RESHAPE variable=edge_weights_table cyclic factor=reshape_factor dim=1
  bool initialized = false;
#else
  static typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_bitwidth];
  static bool initialized = false;
#endif
  if (!initialized) {
    initialize_edge_weights_table<CONFIG_T>(edge_weights_table);
    initialized = true;
  }

  return get_edge_weight<CONFIG_T>(distance, edge_weights_table);
}

template<class data_T, class nvtx_T, class CONFIG_T>
void
compute_features_weights(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  nvtx_T nvtx,
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_propagate],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_propagate],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t aggregation_sums[CONFIG_T::n_aggregators * CONFIG_T::n_propagate]
)
{
  #pragma HLS EXPRESSION_BALANCE

 VerticesCompute:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE
    
    if (iv >= nvtx)
      break;
    
    typename CONFIG_T::index_t const data_offset = iv * CONFIG_T::n_in_features;
    typename CONFIG_T::index_t const features_offset = iv * CONFIG_T::n_propagate;
    typename CONFIG_T::index_t const weights_offset = iv * CONFIG_T::n_aggregators;

    typename CONFIG_T::accum_t propagated_features[CONFIG_T::n_propagate];

    // keras Dense applies weights as K.dot(inputs, kernel) -> kernel is channels first

  Propagate:
    for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
      propagated_features[ip] = input_transform_biases[ip];
    PropagateMatMul:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t data_index = data_offset + ix;
        typename CONFIG_T::index_t weight_index = ix * CONFIG_T::n_propagate + ip;
        propagated_features[ip] += data[data_index] * input_transform_weights[weight_index];
      }
    }

  Accum:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::accum_t distance = aggregator_distance_biases[ia];
    AccumMatMul:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t data_index = data_offset + ix;
        typename CONFIG_T::index_t weight_index = ix * CONFIG_T::n_aggregators + ia;
        distance += data[data_index] * aggregator_distance_weights[weight_index];
      }

      typename CONFIG_T::edge_weight_t edge_weight = compute_garnet_edge_weight<CONFIG_T>(distance);

      edge_weights[weights_offset + ia] = edge_weight;

    AccumPropagate:
      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        typename CONFIG_T::aggr_t cache = edge_weight * propagated_features[ip];
        aggregation_sums[ia * CONFIG_T::n_propagate + ip] += cache;
      }
    }
  }
}

template<class data_T, class nvtx_T, class CONFIG_T>
void
compute_features_weights_collapsed(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  nvtx_T nvtx,
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_propagate],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_propagate],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t aggregation_sums[CONFIG_T::n_aggregators * CONFIG_T::n_propagate]
)
{
  // This is a near-identical function to compute_features_weights. Could not find a way to cleanly factorize while keeping the implementation
  // simple enough for HLS to not get confused.
  
  #pragma HLS EXPRESSION_BALANCE

 VerticesCompute:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE
    
    if (iv >= nvtx)
      break;
    
    typename CONFIG_T::index_t const data_offset = iv * CONFIG_T::n_in_features;
    typename CONFIG_T::index_t const features_offset = iv * CONFIG_T::n_propagate;
    typename CONFIG_T::index_t const weights_offset = iv * CONFIG_T::n_aggregators;

    typename CONFIG_T::accum_t propagated_features[CONFIG_T::n_propagate];

    // keras Dense applies weights as K.dot(inputs, kernel) -> kernel is channels first

  Propagate:
    for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
      propagated_features[ip] = input_transform_biases[ip];
    PropagateMatMul:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t data_index = data_offset + ix;
        typename CONFIG_T::index_t weight_index = ix * CONFIG_T::n_propagate + ip;
        propagated_features[ip] += data[data_index] * input_transform_weights[weight_index];
      }
    }

  Accum:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::accum_t distance = aggregator_distance_biases[ia];
    AccumMatMul:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t data_index = data_offset + ix;
        typename CONFIG_T::index_t weight_index = ix * CONFIG_T::n_aggregators + ia;
        distance += data[data_index] * aggregator_distance_weights[weight_index];
      }

      typename CONFIG_T::edge_weight_t edge_weight = compute_garnet_edge_weight<CONFIG_T>(distance);

      edge_weight_sums[ia] += edge_weight;

    AccumPropagate:
      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        typename CONFIG_T::aggr_t cache = edge_weight * propagated_features[ip];
        aggregation_sums[ia * CONFIG_T::n_propagate + ip] += cache;
      }
    }
  }
}

template<class res_T, class nvtx_T, class CONFIG_T>
void
set_output(
  nvtx_T nvtx,
  typename CONFIG_T::accum_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregation_sums[CONFIG_T::n_aggregators * CONFIG_T::n_propagate],
  typename CONFIG_T::output_transform_weights_t const output_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_propagate * CONFIG_T::n_filters],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_filters],
  res_T res[CONFIG_T::n_vertices * CONFIG_T::n_filters]
)
{ 
  #pragma HLS PIPELINE
  #pragma HLS EXPRESSION_BALANCE

  typename CONFIG_T::aggr_t const vnorm = 1. / CONFIG_T::n_vertices;

 AggrSumNormalize:
  for (unsigned il = 0; il < n_latent; ++il) {
    aggregation_sums[il] *= vnorm;
  }

 Output:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    if (iv >= nvtx)
      break;

    typename CONFIG_T::index_t res_offset = iv * CONFIG_T::n_filters;
    typename CONFIG_T::index_t edge_weight_offset = iv * CONFIG_T::n_aggregators;

  OutputFilt:
    for (unsigned io = 0; io < CONFIG_T::n_filters; ++io) {
      typename CONFIG_T::aggr_t aggr = output_transform_biases[io];

    OutputAggr:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::edge_weight_t edge_weight = edge_weights[edge_weight_offset + ia];

      OutputProp:
        for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
          typename CONFIG_T::index_t il = ia * CONFIG_T::n_propagate + ip;
          typename CONFIG_T::index_t weight_index = il * CONFIG_T::n_filters + io;
        
          aggr += edge_weight * aggregation_sums[il] * output_transform_weights[weight_index];
        }
      }

      res[res_offset + io] = aggr;
    }
  }
}

template<class res_T, class nvtx_T, class CONFIG_T>
void
set_output_collapsed(
  nvtx_T nvtx,
  typename CONFIG_T::aggr_t const edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregation_sums[CONFIG_T::n_aggregators * CONFIG_T::n_propagate],
  typename CONFIG_T::output_transform_weights_t const output_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_propagate * CONFIG_T::n_filters],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_filters],
  res_T res[CONFIG_T::n_filters],
  int collapse_type
)
{
  #pragma HLS PIPELINE
  #pragma HLS EXPRESSION_BALANCE

  typename CONFIG_T::aggr_t const vnorm2 = 1. / CONFIG_T::n_vertices / CONFIG_T::n_vertices;
  typename CONFIG_T::aggr_t const nvtx_vnorm = float(nvtx) / CONFIG_T::n_vertices;

 Output:
  for (int io = 0; io < CONFIG_T::n_filters; ++io) {
    typename CONFIG_T::aggr_t aggr = 0.;

  OutputAggr:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    OutputProp:
      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        typename CONFIG_T::index_t il = ia * CONFIG_T::n_propagate + ip;
        typename CONFIG_T::index_t weight_index = il * CONFIG_T::n_filters + io;
          
        aggr += edge_weight_sums[ia] * aggregation_sums[il] * output_transform_weights[weight_index];
      }
    }

    if (collapse_type == CONFIG_T::collapse_mean) {
      aggr *= vnorm2;
      aggr += output_transform_biases[io] * nvtx_vnorm;
    }
    else if (collapse_type == CONFIG_T::collapse_sum) {
      aggr += output_transform_biases[io] * nvtx;
    }

    res[io] = aggr;
  }
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
  typedef ap_ufixed<64, 32> edge_weight_t;
  typedef ap_fixed<64, 24> aggr_t;

  typedef unsigned short index_t;

  // Layer specs
  static const unsigned n_vertices = 250;
  static const unsigned n_in_features = 4;
  static const unsigned n_aggregators = 4;
  static const unsigned n_filters = 4;
  static const unsigned n_propagate = 4;
  static const unsigned distance_bitwidth = 10;

  // Optimization specs
  static const unsigned reuse_factor = 64;

  enum CollapseType {
    collapse_mean,
    collapse_sum,
    n_collapse_types
  };
};

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
void garnet_passthrough(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_filters],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_propagate],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_propagate],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_weights_t const output_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_propagate * CONFIG_T::n_filters],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_filters]
)
{
  #pragma HLS DATAFLOW
  
  typename CONFIG_T::aggr_t aggregation_sums[CONFIG_T::n_aggregators * CONFIG_T::n_propagate];
  #pragma HLS ARRAY_RESHAPE variable=aggregation_sums complete dim=1

  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];
  unsigned const reshape_factor = CONFIG_T::n_aggregators * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
  #pragma HLS ARRAY_RESHAPE variable=edge_weights cyclic factor=reshape_factor dim=1

  nvtx_T nvtx_local_1 = nvtx[0];
  nvtx_T nvtx_local_2 = nvtx[0];

  aggregation_sums_init<CONFIG_T>(aggregation_sums);

  compute_features_weights<data_T, nvtx_T, CONFIG_T>(
    data,
    nvtx_local_1,
    input_transform_weights,
    input_transform_biases,
    aggregator_distance_weights,
    aggregator_distance_biases,
    edge_weights,
    aggregation_sums
  );

  set_output<res_T, nvtx_T, CONFIG_T>(
    nvtx_local_2,
    edge_weights,
    aggregation_sums,
    output_transform_weights,
    output_transform_biases,
    res
  );
}

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
void garnet_collapsed(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_filters],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_propagate],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_propagate],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_weights_t const output_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_propagate * CONFIG_T::n_filters],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_filters],
    int collapse_type
)
{
  #pragma HLS DATAFLOW
  
  typename CONFIG_T::aggr_t aggregation_sums[CONFIG_T::n_aggregators * CONFIG_T::n_propagate];
  #pragma HLS ARRAY_RESHAPE variable=aggregation_sums complete dim=1

  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=edge_weight_sums complete dim=1

  nvtx_T nvtx_local_1 = nvtx[0];
  nvtx_T nvtx_local_2 = nvtx[0];

  aggregation_sums_init<CONFIG_T>(aggregation_sums);

  edge_weight_sums_init<CONFIG_T>(edge_weight_sums);
  
  compute_features_weights_collapsed<data_T, nvtx_T, CONFIG_T>(
    data,
    nvtx_local_1,
    input_transform_weights,
    input_transform_biases,
    aggregator_distance_weights,
    aggregator_distance_biases,
    edge_weight_sums,
    aggregation_sums
  );

  set_output_collapsed<res_T, nvtx_T, CONFIG_T>(
    nvtx_local_2,
    edge_weight_sums,
    aggregation_sums,
    output_transform_weights,
    output_transform_biases,
    res,
    collapse_type
  );
}

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
void garnet_collapsed_mean(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_filters],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_propagate],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_propagate],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_weights_t const output_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_propagate * CONFIG_T::n_filters],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_filters]
)
{
  garnet_collapsed(
    data, nvtx, res,
    input_transform_weights, input_transform_biases,
    aggregator_distance_weights, aggregator_distance_biases,
    output_transform_weights, output_transform_biases,
    CONFIG_T::collapse_mean
  );
}

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
void garnet_collapsed_sum(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_filters],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_propagate],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_propagate],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_weights_t const output_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_propagate * CONFIG_T::n_filters],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_filters]
)
{
  garnet_collapsed(
    data, nvtx, res,
    input_transform_weights, input_transform_biases,
    aggregator_distance_weights, aggregator_distance_biases,
    output_transform_weights, output_transform_biases,
    CONFIG_T::collapse_sum
  );
}

}

#endif
