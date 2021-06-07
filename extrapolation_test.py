from bias_correction_masked import *
from return_period_storyline import return_period_storyline

def extrapolation_test(df_clim_agg_chosen, df_features_ec_season_us, df_features_ec_season_2C_us, df_features_ec_season_3C_us,
                       brf_model_us, df_clim_2012_us, table_JO_prob2012_us, table_events_prob2012_us,
                       df_joint_or_rf_us, proof_total_us):
                                
    def out_in_range(out_data, in_data, scenario):
        print("MAX",scenario, out_data[out_data > in_data.max(axis=0)].count())
        print("MIN",scenario, out_data[out_data < in_data.min(axis=0)].count())
        
        for feature in in_data:
            out_column = out_data.loc[:,feature]
            in_column = in_data.loc[:,feature]
            
            out_column[out_column > in_column.max()] = in_column.max()
            out_column[out_column < in_column.min()] = in_column.min()
            
            max_count = out_data[out_data > in_data.max(axis=0)].count()
            min_count = out_data[out_data < in_data.min(axis=0)].count()
        
        print("MAX CORRECTED",scenario, out_data[out_data > in_data.max(axis=0)].count())
        print("MIN CORRECTED",scenario, out_data[out_data < in_data.min(axis=0)].count())
        
        return max_count, min_count
        
        # return out_data
    
    
    max_count_PD, min_count_PD = out_in_range(df_features_ec_season_us, df_clim_agg_chosen, scenario = 'PD')
    max_count_2C, min_count_2C = out_in_range(df_features_ec_season_2C_us, df_clim_agg_chosen, scenario = '2C')
    max_count_3C, min_count_3C = out_in_range(df_features_ec_season_3C_us, df_clim_agg_chosen, scenario = '3C')
    
    
    table_scores_us, table_events_prob2012_us = predictions_permutation(brf_model_us, df_clim_agg_chosen, df_features_ec_season_us, df_features_ec_season_2C_us, df_features_ec_season_3C_us, df_clim_2012_us  )
    
    df_joint_or_rf_us, table_JO_prob2012_us = compound_exploration(brf_model_us, df_features_ec_season_us, df_features_ec_season_2C_us, df_features_ec_season_3C_us, df_clim_2012_us )
    
    mean_conditions_similar_2012_2C_us, mean_conditions_similar_2012_3C_us = return_period_storyline(
        df_features_ec_season_us, df_features_ec_season_2C_us, df_clim_agg_chosen,
        table_JO_prob2012_us, table_events_prob2012_us, brf_model_us,
        df_clim_2012_us, df_joint_or_rf_us, proof_total_us, df_features_ec_season_3C_us)
    
    return df_features_ec_season_us, df_features_ec_season_2C_us, df_features_ec_season_3C_us