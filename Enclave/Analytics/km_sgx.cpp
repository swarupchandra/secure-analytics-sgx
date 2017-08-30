#include <stdio.h>      /* vsnprintf */
#include <cstring>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */

#include "analytics_util.h"
#include "kmeans.h"


//E-Step: Compute closest centroid
void closestCentroid(struc_cluster *cls, uint32_t K, double **data, uint32_t num_data, uint32_t num_raw_features) {

    for(int i=0; i<num_data; i++) {
        int min_cls = -1;
        double min_dist = -1;
        //check distance to each cluster centroid
        for(int j=0; j<K; j++) {
            //TODO change objective function if necessary
            double dist = euclidean_distance(cls[j].centroid, data[i], num_raw_features);
            if(min_dist == -1 || dist < min_dist) {
                min_dist = dist;
                min_cls = j;
            }
        }

        // For every data instance, adjust its cluster proportions and cluster radius 
        for(int j=0; j<num_raw_features; j++) {
            cls[min_cls].clus_data_sum[j] += data[i][j];
        }
        ++cls[min_cls].data_len;
        int class_label = (int) data[i][num_raw_features];
        ++cls[min_cls].class_prop[class_label];
        if(min_dist > cls[min_cls].max_data_dist) {
            cls[min_cls].max_data_dist = min_dist;
        }
    }

}



//Main EM algorithm for cluster construction
void EM(double **data, uint32_t num_data, uint32_t raw_num_features, uint32_t K, struc_cluster *cls, uint32_t termination_count) {

    //initialize
    int *random_data = new int[num_data];
    for(int i=0; i<num_data; i++) {
        random_data[i] = i;
    }
    get_random_shuffle(random_data, num_data);
    
    //initialize cluster
    // printf("Selecting random points as centroids.\n");
    for(int i=0; i<K; i++) {
        //set centroid
        for(int j=0; j<raw_num_features; j++) {
            cls[i].centroid[j] = data[random_data[i]][j];
        }
        cls[i].empty = true;
    }

    delete [] random_data;
    
    //Start EM Iteration
    // printf("\nBefore iteration\n");
    // printClusters(cls, K, raw_num_features);

    //start iteration
    int count = 0;
    while(count < termination_count) {
        //E-step = assign points to closest cluster
        closestCentroid(cls, K, data, num_data, raw_num_features);

        //M-step = recompute cluster centroid
        recomputeCentroid(cls, K, data, raw_num_features);

        //for each cluster, reset data and class prop
        for(int i=0; i<K; i++) {
            if(count < termination_count-1) {
                for(int j=0; j<cls[i].num_classes; j++) {
                    cls[i].class_prop[j] = 0;
                }
                for(int j=0; j<raw_num_features; j++){
                    cls[i].clus_data_sum[j] = 0;
                }
                cls[i].data_len = 0;
            }
        }
        ++count;
    }

    //set cluster class to max prop.
    for(int i=0; i<K; i++) {
        int max_class = -1;
        int max_prop = 0;
        for(int j=0; j<cls[i].num_classes; j++) {
            if(cls[i].class_prop[j] > max_prop) {
                max_prop = cls[i].class_prop[j];
                max_class = j;
            }
        }
        cls[i].class_label = max_class;
    }
}


// centroid recomputation for test data 
void recomputeCurrCentroid(struc_cluster *cls, int curr, double *data, uint32_t num_raw_features) {

    // add data to cluster in an oblivious manner, replace latest index + 1;
    for(int i=0; i<num_raw_features; i++) {
        cls[curr].clus_data_sum[i] += (data[i]);
    }
    cls[curr].data_len += 1;
    cls[curr].empty = false;

    // recompute centrold 
    for(int j=0; j<num_raw_features; j++) {
        cls[curr].centroid[j] = (cls[curr].clus_data_sum[j] / cls[curr].data_len);    
    }
    
}


//Predict class label and output size of novel class buffer.
int test(double *data, uint32_t num_features, struc_cluster *cls, uint32_t K) {
        
    //check distance to each cluster and find min
    double min_dist = -1;
    int min_cls = -1;
    
    for(int j=0; j<K; j++) {
        if (cls[j].empty) {
            continue;
        }

        int dist = euclidean_distance(cls[j].centroid, data, num_features-1);
        if(min_dist == -1) {
            min_dist = dist;
            min_cls = j;
        } else if(dist < min_dist) {
            min_dist = dist;
            min_cls = j;
        }
    }

    //check class label
    int res = 0;
    if(cls[min_cls].class_label == (int)data[num_features-1]) {
        // printf("TRUE ");
        res = 1;
    } else {
        // printf("FALSE ");
        res = 0;
    }

    //recompute centroid of min_cls cluster 
    recomputeCurrCentroid(cls, min_cls, data, num_features-1);

    return res;
}


////////////////////////////////////////////////////////////////////////////


struc_cluster *clusters;

//ocall training
void startKMTraining(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t K_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );
    
    uint32_t num_values = 11;
    int num_iteration = (num_data/iter_size - 1); //num of chunks
    int K = K_size;

    uint32_t  termination_count = 10; //EM

    //GET NEW DATA
    uint32_t start_data = 0;
    uint32_t datasize = iter_size; //200;

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
    clusters = new struc_cluster[num_classes*5];

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
    printf("No. of features = %d\n", num_features);

    //INITIAL CLUSTER

    printf("Initializing clusters\n");
    for(int i=0; i<K; i++) {
        //set centroid features, data, class proportions to -1
        clusters[i].cluster_num = i;
        clusters[i].centroid = new double[num_features-1];
        clusters[i].clus_data_sum = new double[num_features-1];
        for(int j=0; j<num_features-1; j++) {
            clusters[i].centroid[j] = -1;
            clusters[i].clus_data_sum[j] = 0;
        }
        
        clusters[i].data_len = 0;
        clusters[i].class_prop = new int[num_classes];
        for(int j=0; j<num_classes; j++) {
            clusters[i].class_prop[j] = 0;
        }
        clusters[i].num_classes = num_classes;
        clusters[i].class_label = -1;
        clusters[i].max_data_dist = 0;
        clusters[i].empty = true;
    }
    
    //TRAIN DECISION TREE;
    printf("Perform EM clustering.\n");
    EM(data, datasize, num_features-1, K, clusters, termination_count);
    printf("Clusters completed learned.\n");
    
    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] s;
}


//ocall testing
void startKMTesting(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t K_size) {
    
    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    uint32_t num_values = 11;
    int num_iteration = (num_data/iter_size - 1); //num of chunks
    int K = K_size;

    uint32_t  termination_count = 10; //EM

    //GET NEW DATA
    uint32_t start_data = 0;
    uint32_t datasize = iter_size; //200;

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

    //START STREAMING
    int iteration_count = 0;
    int correct = 0;
    int total_pred = 0;
    double acc = 0;

    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*100;
    char *s = new char[rlen];
    char *res_out = new char[datasize];

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
        printf("No. of features = %d\n", num_features);

        
        // TEST NEW DATA
        printf("Testing iteration ...\n");
        
        for(int i=0; i<datasize; i++) {
            int res = test(data[i], num_features, clusters, K);
            if(res > 0) {
                correct += 1;   
            }
            res_out[i] = (char)res;
            total_pred += 1;
        }

        // encrypt_data(res_out, datasize);

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
    
    
    //delete clusters
    for(int i=0; i<K; i++) {
        delete [] clusters[i].centroid;
        delete [] clusters[i].class_prop;
        delete [] clusters[i].clus_data_sum;
    }
    delete [] clusters;


    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] s;
    delete [] res_out;

    printf("END OF ENCLAVE STREAM MINING.\n");
    ocall_print_acc(acc);
}
