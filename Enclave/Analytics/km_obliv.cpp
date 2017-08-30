#include <stdio.h>      /* vsnprintf */
#include <cstring>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */
#include "sgx_tcrypto.h"

#include "analytics_util.h"
#include "kmeans.h"


//E-Step: Compute closest centroid
void obliv_closestCentroid(struc_cluster *cls, uint32_t K, double **data, uint32_t num_data, uint32_t num_raw_features) {

    for(int i=0; i<num_data; i++) {
        int min_cls = -1;
        double min_dist = -1;
        //check distance to each cluster centroid
        for(int j=0; j<K; j++) {
            //TODO change objective function
            double dist = euclidean_distance(cls[j].centroid, data[i], num_raw_features);
            //printf("Distance for dist=%f\n", dist);
            int decoy_if;
            if(min_dist == -1 ) {
                decoy_if = 1;
            } else {
                if(min_dist > dist) {
                    decoy_if = 1;
                } else {
                    //decoy
                    decoy_if = 0;
                }
            }
            min_dist = (dist*decoy_if + min_dist*(1-decoy_if));
            min_cls = (j*decoy_if + min_cls*(1-decoy_if));
        }

        // For every data instance, adjust its cluster proportions and cluster radius 
        
        for(int j=0; j<K; j++) {
            int decoy_min_cls = oequal_int(min_cls, j);
            // if(min_cls == j) {
            //     decoy_min_cls = 1;
            // } else {
            //     decoy_min_cls = 0;
            // }
            
            for(int p=0; p<num_raw_features; p++) {
                cls[j].clus_data_sum[p] += data[i][p]*decoy_min_cls;
            }
            cls[j].data_len += decoy_min_cls;

            int class_label = (int) data[i][num_raw_features];
            for(int p=0; p<cls[j].num_classes; p++) {
                int decoy_label = oequal_int(p, class_label);
                // if(p == class_label) {
                //     decoy_label = 1;
                // } else {
                //     decoy_label = 0;
                // }
                cls[j].class_prop[p] += (decoy_min_cls*decoy_label) + ((1-decoy_label)*decoy_min_cls);
            }
            
            cls[j].max_data_dist = omax_double((min_dist*decoy_min_cls), cls[j].max_data_dist);
        }
    }

}


// M-Step: Recompute cluster centroid
void obliv_recomputeCentroid(struc_cluster *cls, uint32_t K, double **data, int num_data, uint32_t num_raw_features) {
    //for each cluster
    for(int i=0; i<K; i++) {        
        for(int j=0; j<num_raw_features; j++) {            
            cls[i].centroid[j] = 0;

            int decoy_if = ogreater_int(cls[i].data_len, 0);
            if(cls[i].data_len > 0) {
                // decoy_if = 1;
                // decoy_denom = (decoy_denom*0 + cls[i].data_len*1);
                cls[i].empty = false;
            } else {
                // decoy_if = 0;
                // decoy_denom = (decoy_denom*1 + cls[i].data_len*0);
                cls[i].empty = true;
            }
            int decoy_denom = (1-decoy_if) + (cls[i].data_len*decoy_if);
            cls[i].centroid[j] = cls[i].clus_data_sum[j] / decoy_denom;
        }
    }    
}



// //Pseudo random number generation to select initial centroid
// void nonobli_random_shuffle(int *num, int num_data) {

//     uint32_t val;
//     sgx_read_rand((unsigned char *)&val, 4);

//     //Fisher–Yates shuffle algorithm
//     for(int i=num_data-1; i>0; i-- ){
//         int r = (uint32_t)val%i;
//         int temp = num[i];
//         num[i] = num[r];
//         num[r] = temp;
//     }
// }

//DECOY:
//1. defer E-step assignment after finding min-distance cluster for every data.
//2. Also capture min distance for corresponding cluster

//Main EM algorithm for cluster construction
void obliv_EM(double **data, uint32_t num_data, uint32_t raw_num_features, uint32_t num_classes, uint32_t K, struc_cluster *cls, uint32_t termination_count) {

    //initialize
    int *random_data = new int[num_data];
    for(int i=0; i<num_data; i++) {
        random_data[i] = i;
    }
    //Non oblivious
    get_random_shuffle(random_data, num_data);

    //initialize cluster
    // printf("Selecting random points as centroids.\n");
    for(int i=0; i<K; i++) {
        //set centroid
        //printf("Random=%d\n", random[i]);
        
        for(int p=0; p<num_data; p++) {
            int decoy_index = oequal_int(p, random_data[i]);
            // if(p == random_data[i]) {
            //     decoy_index = 1;
            // } else {
            //     decoy_index = 0;
            // }

            for(int j=0; j<raw_num_features; j++) {
                cls[i].centroid[j] = (data[p][j]*decoy_index) + ((1-decoy_index)*cls[i].centroid[j]);
            }
        }
        
        cls[i].empty = true;
    }

    delete [] random_data;

    
    //Start EM Iteration
    // printf("\nBefore iteration\n");
    // printoclusters(cls, K, raw_num_features);
    
    // printf("Begin iteration\n");
    //start iteration
    int count = 0;
    while(count < termination_count) {
        //E-step = assign points to closest cluster
        // printf("E-Step : Assigning data to centroids\n");
        obliv_closestCentroid(cls, K, data, num_data, raw_num_features);

        //M-step = recompute cluster centroid
        // printf("M-Step : Recomputing centroid\n");
        obliv_recomputeCentroid(cls, K, data, num_data, raw_num_features);

        //for each cluster, reset data and class prop
        for(int i=0; i<K; i++) {
            if(count < termination_count-1) {
                for(int j=0; j<cls[i].num_classes; j++) {
                    cls[i].class_prop[j] = 0;
                }

                // for(int j=0; j<num_data; j++) {
                //     cls[i].data_index[j] = -1;
                // }
                for(int j=0; j<raw_num_features; j++){
                    cls[i].clus_data_sum[j] = 0;
                }
                cls[i].data_len = 0;
            }
        }
        ++count;
    }
    
    // printf("After iteration\n");
    // printoclusters(cls, K, raw_num_features);

     //compute class proportions
    //  for(int i=0; i<K; i++) {
    //     for(int j=0; j<cls[i].num_classes; j++) {
    //         cls[i].class_prop[j] = 0;
    //     }
    //     for(int j=0; j<cls[i].data_len; j++) {
    //         int index = cls[i].data_index[j];
    //         int decoy_index;
    //         int class_label = 0;
    //         for(int q=0; q<num_data; q++) {
    //             if(q == index) {
    //                 decoy_index = 1;
    //             } else {
    //                 decoy_index = 0;
    //             }
    //             class_label = (data[q][raw_num_features]*decoy_index) + ((1-decoy_index)*class_label);
    //         }
            
    //         for(int p=0; p<cls[i].num_classes; p++) {
    //             int decoy_if;
    //             if(p == class_label) {
    //                 decoy_if = 1;
    //             } else {
    //                 decoy_if = 0;
    //             }
    //             cls[i].class_prop[p] += decoy_if;
    //         }
            
    //     }
    // }

    //set cluster class to max prop.
    //TODO oblivious max
    for(int i=0; i<K; i++) {
        int max_class = -1;
        int max_prop = 0;
        for(int j=0; j<cls[i].num_classes; j++) {
            int decoy_if = ogreater_int(cls[i].class_prop[j], max_prop);
            // if(cls[i].class_prop[j] > max_prop) {
            //     decoy_if = 1;
            //     // max_prop = (cls[i].class_prop[j]*1 + max_prop*0);
            //     // max_class = (j*1 + max_class*0);
            // } else {
            //     //decoy
            //     decoy_if = 0;
            //     // max_prop = (cls[i].class_prop[j]*0 + max_prop*1);
            //     // max_class = (j*0 + max_class*1);
            // }
            max_prop = (cls[i].class_prop[j]*decoy_if) + (max_prop*(1-decoy_if));
            max_class = (j*decoy_if) + (max_class*(1-decoy_if));
        }
        cls[i].class_label = max_class;
    }
}


// Data oblivious centroid recomputation for test data 
void obliv_recomputeCurrCentroid(struc_cluster *cls, int curr, double *data, int row, uint32_t num_raw_features) {
    // printf("recomputing current centroid\n");
    // add data to cluster in an oblivious manner, replace latest index + 1;
    
    for(int i=0; i<num_raw_features; i++) {
        cls[curr].clus_data_sum[i] += (data[i]*row);
    }
    cls[curr].data_len += row;
    cls[curr].empty = false;
    
    // printf("new data added\n");

    // recompute centrold 
    for(int j=0; j<num_raw_features; j++) {
        cls[curr].centroid[j] = (cls[curr].clus_data_sum[j] / cls[curr].data_len);    
    }
}


//Predict class label and output size of novel class buffer.
int obliv_test(double *data, uint32_t num_features, struc_cluster *cls, uint32_t K) {
        
    //check distance to each cluster and find min
    double min_dist = -1;
    int min_cls = -1;

    for(int j=0; j<K; j++) {

        if (cls[j].empty) {
            continue;
        }

        int dist = euclidean_distance(cls[j].centroid, data, num_features-1);
        int decoy_if;
        if(min_dist == -1) {
            decoy_if = 1;
        } else if(dist < min_dist) {
            decoy_if = 1;
        } else {
            //decoy
            decoy_if = 0;
        }
        min_dist = (dist*decoy_if) + (min_dist*(1-decoy_if));
        min_cls = (j*decoy_if) + (min_cls*(1-decoy_if));
    }

    int res = 0;
    for(int j=0; j<K; j++) {
        int decoy_result = oequal_int(j, min_cls);
        // if(j == min_cls) {
        //     decoy_result = 1;
        // } else {
        //     decoy_result = 0;
        // }

        int decoy_class_if = oequal_int(cls[j].class_label, (int)data[num_features-1]);
        // if(cls[j].class_label == (int)data[num_features-1]) {
        //     // printf("TRUE ");
        //     decoy_class_if = 1;
        // } else {
        //     // printf("FALSE ");
        //     decoy_class_if = 0;
        // }   
        res = omax_int(decoy_result & decoy_class_if, res);

        //recompute centroid of min_cls cluster 
        obliv_recomputeCurrCentroid(cls, j, data, decoy_result, num_features-1);
    }

    // printf("Done testing.\n");

    return res;

}




// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%% MAIN PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

struc_cluster *oclusters;

//ocall training
void startKMOblivTraining(int num_data, int num_features, int num_classes, int iter_size, int K_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    uint32_t num_values = 11;
    int num_iteration = (num_data/iter_size - 1); //num of chunks
    int K = K_size;
    uint32_t  termination_count = 10; //EM

    //GET NEW DATA
    uint32_t start_data = 0;
    int datasize = iter_size; //200;

    uint32_t len = sizeof(char)*datasize*num_features*100;
    // char data_str[len];

    //incoming data buffer
    double **data = new double*[datasize];
    for(int i=0; i<datasize; i++) {
        data[i] = new double[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = -1;
        }
    }

    //MODEL PARAMETERS

    oclusters = new struc_cluster[num_classes*5];

    //START STREAMING
    int iteration_count = 0;
    int correct = 0;
    int total_pred = 0;
    double acc = 0;

    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*10;
    char *s = new char[rlen];

    printf("Reading data from app.\n");
    int row_count = 0;
    for(int i=start_data; i<(start_data+datasize); i++) {
        ocall_read_data(i, 1, s, rlen);
        // decrypt_data(s, rlen)
        extract_features(row_count, data, s, datasize, num_features);
        ++row_count;
        
    }
    printf("row_count = %d\n", row_count);
    printf("No. of features = %d\n" , num_features);

    //shuffle data instances
    randomize_rand_index(data, datasize, num_features);

    // printf("Initializing oclusters\n");
    for(int i=0; i<K; i++) {
        //set centroid
        oclusters[i].cluster_num = i;
        oclusters[i].centroid = new double[num_features-1];
        oclusters[i].clus_data_sum = new double[num_features-1];
        for(int j=0; j<num_features-1; j++) {
            oclusters[i].centroid[j] = -1;
            oclusters[i].clus_data_sum[j] = 0;
        }

        oclusters[i].data_len = 0;
        oclusters[i].class_prop = new int[num_classes];
        for(int j=0; j<num_classes; j++) {
            oclusters[i].class_prop[j] = 0;
        }
        oclusters[i].num_classes = num_classes;
        oclusters[i].class_label = -1;
        oclusters[i].max_data_dist = 0;
        oclusters[i].empty = true;
    }
    
    //TRAIN DECISION TREE;
    printf("Perform EM clustering.\n");
    obliv_EM(data, datasize, num_features-1, num_classes, K, oclusters, termination_count);
    printf("oclusters learning completed.\n");
    
    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] s;
}


//ocall testing
void startKMOblivTesting(int num_data, int num_features, int num_classes, int iter_size, int K_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    uint32_t num_values = 11;
    int num_iteration = (num_data/iter_size - 1); //num of chunks
    int K = K_size;
    uint32_t  termination_count = 10; //EM

    //GET NEW DATA
    uint32_t start_data = 0;
    int datasize = iter_size; //200;

    uint32_t len = sizeof(char)*datasize*num_features*100;

    //incoming data buffer
    double **data = new double*[datasize];
    for(int i=0; i<datasize; i++) {
        data[i] = new double[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = -1;
        }
    }

    //MODEL PARAMETERS


    //START STREAMING
    int iteration_count = 0;
    int correct = 0;
    int total_pred = 0;
    double acc = 0;

    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*10;
    char *s = new char[rlen];

    do {
        printf("Reading data from app.\n");
        int row_count = 0;
        for(int i=start_data; i<(start_data+datasize); i++) {
            ocall_read_data(i, 1, s, rlen);
            // decrypt_data(s, rlen)
            extract_features(row_count, data, s, datasize, num_features);
            ++row_count;
            
        }
        printf("row_count = %d\n", row_count);
        printf("No. of features = %d\n" , num_features);

        //shuffle data instances
        randomize_rand_index(data, datasize, num_features);

        //TEST NEW DATA and UPDATE CPD BUFFER
        printf("Testing iteration ...\n");
        char *res_out = new char[datasize];
        for(int i=0; i<datasize; i++) {
            // printf("Testing %d ", i);
            int res = obliv_test(data[i], num_features, oclusters, K);
            int decoy_if_correct, decoy_if_total;
            if(res > 0) {
                decoy_if_correct = 1;
                decoy_if_total = 1;
            } else if(res < 0) {
                //outlier
                decoy_if_correct = 0;
                decoy_if_total = 1;
            } else {
                decoy_if_correct = 0;
                decoy_if_total = 1;
            }
            correct += decoy_if_correct;
            total_pred += decoy_if_total;
            res_out[i] = (char)res;
        }

        // encrypt_data(res_out, datasize);
        delete [] res_out;

        if(total_pred > 0) {
            acc = (double) correct / total_pred;
            printf("Acc=%f\n", acc);
        }
        

        //NEXT ITERATION
        start_data += datasize;
        for(int i=0; i<datasize; i++) {
            for(int j=0; j<num_features; j++) {
                data[i][j] = -1;
            }
        }

        if(iteration_count == num_iteration) {
            printf("END\n");
            break;
        }
        
        ++iteration_count;
        printf("******************* NEXT ITERATION (%d, %d)***************************\n", start_data, (start_data + datasize));

    } while (start_data < num_data);
    
    
    //delete oclusters
    for(int i=0; i<K; i++) {
        delete [] oclusters[i].centroid;
        delete [] oclusters[i].clus_data_sum;
        delete [] oclusters[i].class_prop;
    }
    delete [] oclusters;

    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] s;

    printf("END OF ENCLAVE STREAM TESTING.\n");
    ocall_print_acc(acc);
}

