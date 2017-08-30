
#include "analytics_util.h"

// TODO: Many need to change the way dummy data is generated 
void generate_new_data(int **data, bool *rows, int num_data, int num_features) {
    //for each feature, get max and min value.
    //uniformly generate random values within bounds
    for(int i=0; i<num_features-1; i++) {
        int max = -1;
        int min = -1;
        //find max and min value for each feature in the dataset
        for(int j=0; j<num_data; j++) {
            int decoy_check = oblivious_check(rows, j);
            if(max == -1) {
                max = data[j][i]*decoy_check + max*(1-decoy_check);
                min = max;
                continue;
            }

            if(data[j][i] > max) {
                max = data[j][i]*decoy_check + max*(1-decoy_check);
            }
            if(data[j][i] < min) {
                min = data[j][i]*decoy_check + min*(1-decoy_check);
            }
            
        }
        
        for(int j=0; j<num_data; j++) {
            int decoy_check = oblivious_check(rows, j);

            //randomly generate number between min and max for dummy data
            int r = 0;
            int decoy_r = ogreater_int(max, 0);
            r = get_random_int((decoy_r*max) + (1-decoy_r));
            data[j][i] = (r*(1-decoy_check)) + (data[j][i]*decoy_check);
        }
    }

}

void generate_new_data(double **data, bool *rows, int num_data, int num_features) {
    //for each feature, get max and min value.
    //uniformly generate random values within bounds
    for(int i=0; i<num_features-1; i++) {
        double max = -1;
        double min = -1;
        //find max and min value for each feature in the dataset
        for(int j=0; j<num_data; j++) {
            if(max == -1) {
                max = data[j][i];
                min = max;
                continue;
            }

            int decoy_min_max = ogreater_int(data[j][i], max);
            max = data[j][i]*decoy_min_max + max*(1-decoy_min_max);

            decoy_min_max = ogreater_int(min, data[j][i]);
            min = data[j][i]*decoy_min_max + min*(1-decoy_min_max);
        }

        // printf("Max=%f, Min=%f\n", max, min);
        for(int j=0; j<num_data; j++) {
            int decoy_if = oblivious_check(rows, j);
            //randomly generate number between min and max for dummy data
            if(i < num_features-1) {
                double r = 0;
                int decoy_r = 1 - oequal_int(max, 0);
                r = get_random_double((decoy_r*max) + (1-decoy_r), min);
                data[j][i] = (r*(1-decoy_if)) + (data[j][i]*decoy_if);
            } else {
                int r = get_random_int(max);
                data[j][i] = (r*(1-decoy_if)) + (data[j][i]*decoy_if);
            }
            
        }
    }

}

// change feature values and class labels for decoy data (random indices - globally stored)
void change_decoy_features(int **data, bool *rows, int num_data, int num_features, int num_classes) {
    //assign a random value to each feature to fake data 
    generate_new_data(data, rows, num_data, num_features);

    // printf("Generated new features\n");
    for(int i=0; i<num_data; i++) {
        //assign a random class label to fake data 
        int decoy_class_label = get_random_int(num_classes);
        int decoy_check = oblivious_check(rows, i);
        data[i][num_features-1] = (decoy_class_label*(1-decoy_check)) + (data[i][num_features-1]*decoy_check);
    }
}

// creating new fake data 
void add_decoy_data(int **data, int num_data, int fake_times, int num_features, int num_classes, bool *rows) {
    //randomize data positions
    randomize_data_index(data, num_data*fake_times, num_features);

    //update decoy flag on original and fake data.
    for(int i=0; i<num_data*fake_times; i++) {
        if(data[i][num_features-1] < 0) {
            rows[i] = false;
        } else {
            rows[i] = true;
        }
    }

    // change labels of fake data 
    change_decoy_features(data, rows, num_data*fake_times, num_features, num_classes);
}

// creating new fake data 
void add_decoy_data(double **data, int num_data, int fake_times, int num_features, int num_classes, bool *rows) {
    //randomize data positions
    randomize_rand_index(data, num_data*fake_times, num_features);

    //update decoy flag on original and fake data.
    for(int i=0; i<num_data*fake_times; i++) {
        if(data[i][num_features-1] < 0) {
            rows[i] = false;
        } else {
            rows[i] = true;
        }
    }
    
    // change features and labels of fake data 
    generate_new_data(data, rows, num_data*fake_times, num_features);

}

