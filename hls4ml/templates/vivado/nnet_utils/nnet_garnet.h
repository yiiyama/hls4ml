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

    // Layer Sizes
    static const unsigned n_in_hits = 250;
    static const unsigned n_in_features = 4;
    //static const unsigned n_out_hits = 250;
    //static const unsigned n_out_features = 4;
    //static const unsigned n_out = 10;
    static const unsigned n_aggregators = 4;
    static const unsigned n_filters = 4;
    static const unsigned n_propagate = 4;

    //static const unisgned n_tc =10

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
};
 template<class data_T, class res_T, typename CONFIG_T>
void garnet(
    data_T    data[CONFIG_T::n_in_hits * CONFIG_T::n_in_features],
    res_T     res[CONFIG_T::n_in_hits * CONFIG_T::n_filters],
    //data_T    data_aggregators[CONFIG_T::n_aggregators],
    //data_T    data_filters[CONFIG_T::n_filters],
    //data_T    data_propogate[CONFIG_T::n_propagate],
    typename CONFIG_T::weight_t  input_transform_weights[CONFIG_T::n_in_hits * CONFIG_T::n_in_features * CONFIG_T::n_propagate],
    typename CONFIG_T::bias_t    input_transform_biases[CONFIG_T::n_in_hits * CONFIG_T::n_propagate],
    typename CONFIG_T::weight_t  aggregator_distance_weights[CONFIG_T::n_in_hits * CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::bias_t    aggregator_distance_biases[CONFIG_T::n_in_hits * CONFIG_T::n_aggregators],
    typename CONFIG_T::weight_t  output_transform_weights[CONFIG_T::n_in_hits * (CONFIG_T::n_in_features + 2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators+CONFIG_T::n_propagate) + CONFIG_T::n_aggregators) * CONFIG_T:: n_filters],
    typename CONFIG_T::bias_t    output_transform_biases[CONFIG_T::n_in_hits * CONFIG_T::n_filters]
  )
{



  typename CONFIG_T::accum_t features[CONFIG_T::n_in_hits * CONFIG_T::n_propagate];
  // input feature transform
  for(int iv=0; iv<CONFIG_T::n_in_hits; iv++){
  // getting the features propogate
  typename CONFIG_T::data_T d1[CONFIG_T::n_in_features];
  typename CONFIG_T::res_T r1[CONFIG_T::n_propagate];
  //d1 = data[iv*CONFIG_T::n_in_features : (iv*CONFIG_T::n_in_features + CONFIG_T::n_in_features)];
  //r1 = data[iv*CONFIG_T::n_in_features : (iv*CONFIG_T::n_in_features + CONFIG_T::n_in_features)];
  for(int ij =0; ij < CONFIG_T::n_in_features ; ij++){

  	d1[ij] = data[iv*CONFIG_T::n_in_features + ij];	
  }
  dense_latency<data_T, res_T, CONFIG_T>(d1, r1, input_transform_weights, input_transform_biases);
  for(int ij =0; ij < CONFIG_T::n_propagate; ij++){
    features[iv*CONFIG_T::n_propagate + ij] = r1[ij];
  }
  // r1 has output make another loop to copy one by one
  }


  //aggregatore distance
  typename CONFIG_T::accum_t distance[CONFIG_T::n_in_hits * CONFIG_T::n_aggregators];

  for (int iv=0; iv<CONFIG_T::n_in_hits; iv++){
    //getting distance
    typename CONFIG_T::data_T d2[CONFIG_T::n_in_features];
    typename CONFIG_T::res_T r2[CONFIG_T::n_aggregators];
    //d2 = data[iv*CONFIG_T::n_in_features : (iv*CONFIG_T::n_in_features + CONFIG_T::n_in_features)];
    for(int ij = 0; ij< CONFIG_T::n_in_features ; ij++){
    d2[ij]=data[iv*CONFIG_T::n_in_features + ij];
    }
  dense_latency<data_T, res_T, CONFIG_T>(d2, r2, aggregator_distance_weights, aggregator_distance_biases);
  for(int ij =0; ij < CONFIG_T::n_aggregators; ij++){
    distance[iv*CONFIG_T::n_aggregators + ij] = r2[ij];
  }
  }


  //gaussin basicall exponential edge weight,  right now 2 raise to -x
  typename CONFIG_T::accum_t edge_weights[CONFIG_T::n_in_hits * CONFIG_T::n_aggregators];

  for (int iv=0; iv<CONFIG_T::n_in_hits; iv++){
  for(int ij=0; ij<CONFIG_T::n_aggregators; ij++){
    edge_weights[ij+(CONFIG_T::n_aggregators * iv)] = pow(2,-distance[ij+(CONFIG_T::n_aggregators * iv)]);
    }
  }

  //concat edge weight and input feature transfrom (B V F+S)
  typename CONFIG_T::accum_t new_feature[CONFIG_T::n_in_hits * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)];

  for(int iv=0; iv<CONFIG_T::n_in_hits; iv++){
    for(int ij=0; ij<(CONFIG_T::n_propagate + CONFIG_T::n_aggregators); ij++){
      if(ij < CONFIG_T::n_propagate){
        new_feature[(iv * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)) + ij] = features[(iv*CONFIG_T::n_propagate) + ij];
      }
      else{
        new_feature[(iv * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)) + ij] = edge_weights[(iv*CONFIG_T::n_aggregators) + ij - CONFIG_T::n_propagate];
      }
    }
  }

  //transpose edge weights
  // typename CONFIG_T::accum_t edge_weights_trans[CONFIG_T::n_aggregators * CONFIG_T::n_in_hits];
  // for (int iv = 0; iv < CONFIG_T::n_in_hits; iv++){
  //   for (int if = 0; if < CONFIG_T::n_aggregators; if++){
  //     edge_weights_trans[iv] =
  //   }
  // }

  //not necessary lets see
  float mean=0;
  float max = 0;
  typename CONFIG_T:: accum_t common_agg[CONFIG_T::n_in_hits * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators) * CONFIG_T::n_aggregators];
  for(int iv=0; iv < CONFIG_T::n_in_hits; iv++){
    for(int ij=0; ij < CONFIG_T::n_aggregators; ij++){
      for(int ia=0; ia < (CONFIG_T::n_propagate + CONFIG_T::n_aggregators); ia++){
        common_agg[(iv*((CONFIG_T::n_propagate + CONFIG_T::n_aggregators) * CONFIG_T::n_aggregators)) + (ij * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)) + ia] = edge_weights[(iv*CONFIG_T::n_aggregators)+ia] * new_feature[(iv * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)) + ij];
        mean +=  edge_weights[(iv*CONFIG_T::n_aggregators)+ia] * new_feature[(iv * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)) + ij];
        if(max < (edge_weights[(iv*CONFIG_T::n_aggregators)+ia] * new_feature[(iv * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)) + ij])){
          max = edge_weights[(iv*CONFIG_T::n_aggregators)+ia] * new_feature[(iv * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)) + ij];
        }
      }
    }
  }
  mean = mean / CONFIG_T::n_in_hits;

  //aggregate max
  typename CONFIG_T::accum_t agg_max[(CONFIG_T::n_propagate + CONFIG_T::n_aggregators) * CONFIG_T::n_aggregators];
  for(int iv=0; iv < ((CONFIG_T::n_propagate + CONFIG_T::n_aggregators) * CONFIG_T::n_aggregators); iv++){
    float sum=0;
    for (int ij=0; ij < CONFIG_T::n_in_hits; ij++){
      sum+= common_agg[ij*((CONFIG_T::n_propagate + CONFIG_T::n_aggregators) * CONFIG_T::n_aggregators) + iv];
    }
    agg_max[iv] = sum/max;
  }

  //aggregate mean
  typename CONFIG_T::accum_t agg_mean[(CONFIG_T::n_propagate + CONFIG_T::n_aggregators) * CONFIG_T::n_aggregators];
  for(int iv=0; iv < ((CONFIG_T::n_propagate + CONFIG_T::n_aggregators) * CONFIG_T::n_aggregators); iv++){
    float sum=0;
    for (int ij=0; ij < CONFIG_T::n_in_hits; ij++){
      sum+= common_agg[ij*((CONFIG_T::n_propagate + CONFIG_T::n_aggregators) * CONFIG_T::n_aggregators) + iv];
    }
    agg_mean[iv] = sum/mean;
  }
  //concant agg max and agg mean (B S 2*(F+S))
  typename CONFIG_T::accum_t agg_mean_max[2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate)];
  for(int iv=0; iv<CONFIG_T::n_aggregators; iv++){
    for(int ij=0; ij<(2 * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)); ij++){
      if(ij < (CONFIG_T::n_propagate + CONFIG_T::n_aggregators)){
        agg_mean_max[(iv * (2 * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators))) + ij] = agg_max[(iv*CONFIG_T::n_propagate) + ij];
      }
      else{
        agg_mean_max[(iv * (2 * (CONFIG_T::n_propagate + CONFIG_T::n_aggregators))) + ij] = agg_mean[(iv*CONFIG_T::n_propagate) + ij - (CONFIG_T::n_propagate+CONFIG_T::n_aggregators)];
      }
    }
  }
  // apply edge weights agg_max_mean and edge weights
  typename CONFIG_T::accum_t upd_features[CONFIG_T::n_in_hits * 2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate)];
  for(int iv=0; iv < CONFIG_T::n_in_hits; iv++){
    for(int ia=0; ia < CONFIG_T::n_aggregators; ia++){
      for(int ij =0; ij < (2 * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate));ij++){
        upd_features[(iv*CONFIG_T::n_in_hits) + ((2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate)))] = agg_mean_max[ia*CONFIG_T::n_aggregators + ij] * edge_weights[iv*CONFIG_T::n_aggregators + ia];
      }
    }
  }
  //Concatenate x, updated feature and edge weight
  typename CONFIG_T::accum_t updated_features[CONFIG_T::n_in_hits * (CONFIG_T::n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate))+ CONFIG_T::n_aggregators)];
  for(int iv=0; iv< CONFIG_T::n_in_hits; iv++){
      for(int ij=0; ij < (CONFIG_T::n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate) + CONFIG_T::n_aggregators));ij++){
        if( ij < CONFIG_T::n_in_features){
          updated_features[iv*(CONFIG_T::n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate))+ CONFIG_T::n_aggregators) + ij] = data[(iv * CONFIG_T::n_in_features) + ij];
        }
        else if (ij >= CONFIG_T::n_in_features && ij < (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate) + CONFIG_T::n_aggregators)){
          updated_features[iv*(CONFIG_T::n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate))+ CONFIG_T::n_aggregators) + ij] = upd_features[(iv * 2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate)) + ij - CONFIG_T::n_in_features];
        }
        else{
          updated_features[iv*(CONFIG_T::n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate))+ CONFIG_T::n_aggregators) + ij] = edge_weights[(iv * CONFIG_T::n_aggregators) + ij - CONFIG_T::n_in_features - ( 2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate))];
        }
      }
  }

  // updated features to dense
  for(int iv=0; iv<CONFIG_T::n_in_hits; iv++){
    typename CONFIG_T::data_T d3[CONFIG_T::n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate))+ CONFIG_T::n_aggregators];
    typename CONFIG_T::res_T r3[CONFIG_T::n_filters];
    for(int ij = 0 ; ij < (CONFIG_T::n_in_featues + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propogate))+CONFIG_T::n_aggregators ); ij++ ){
    d3[ij] = updated_features[iv*(CONFIG_T::n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate)) + CONFIG_T::n_aggregators) + ij];
    }
    //d3 = data[(iv*(CONFIG_T::n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate))+ CONFIG_T::n_aggregators)) : ((iv*(CONFIG_T:n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate))+ CONFIG_T::n_aggregators)) + (n_in_features + (2 * CONFIG_T::n_aggregators * (CONFIG_T::n_aggregators + CONFIG_T::n_propagate))+ CONFIG_T::n_aggregators))];
  dense_latency<data_T, res_T, CONFIG_T>(d3, r3, output_transform_weights, output_transform_biases);
  
}
}
}
#endif
