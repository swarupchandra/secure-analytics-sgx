#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */
#include "sgx_tcrypto.h"

#include "analytics_util.h"
#include "kmeans.h"
#include "rand_primitives.h"



//////////////////////////////////////////////////////////////////////////////////////////////


//E-Step: Compute closest centroid
// Ignore dummy data instances for centroid obliviously
void rand_closestCentroid(struc_cluster *cls, uint32_t K, double **data, bool *rows, uint32_t num_data, uint32_t num_raw_features) {
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

        // cls[min_cls].data_index[cls[min_cls].data_len] = i;
        int decoy_check = oblivious_check(rows, i);
        for(int j=0; j<num_raw_features; j++) {
            cls[min_cls].clus_data_sum[j] += data[i][j]*decoy_check;
        }
        cls[min_cls].data_len += decoy_check;
        int class_label = (int) data[i][num_raw_features];
        cls[min_cls].class_prop[class_label] += decoy_check;
        int decoy_if = ogreater_int(min_dist, cls[min_cls].max_data_dist);
        cls[min_cls].max_data_dist = (min_dist*decoy_if) + ((1-decoy_if)*cls[min_cls].max_data_dist);
    }

}

//DECOY:
//1. defer E-step assignment after finding min-distance cluster for every data.
//2. Also capture min distance for corresponding cluster

/*
 Changes to original algorithm:
 1. Do not ignore dummy data instances when computing K initial centroid. Simply select first K instances from random shuffle.
 2. Ignore dummy data instances while computing cluster centroid in each cluster (M step).
*/

//Main EM algorithm for cluster construction
void rand_EM(double **data, bool *rows, uint32_t num_data, uint32_t raw_num_features, uint32_t K, struc_cluster *cls, uint32_t termination_count) {   
    // initialize cluster centroid with randomly selected data instances 
    // Obliviously select data 
    int *random_data = new int[num_data];
    for(int i=0; i<num_data; i++) {
        random_data[i] = i;
    }
    get_random_shuffle(random_data, num_data);

    //initialize cluster
    for(int i=0; i<K; i++) {
        //set centroid
        for(int j=0; j<raw_num_features; j++) {
            cls[i].centroid[j] = data[random_data[i]][j];
        }
        cls[i].empty = true;
    }

    delete [] random_data;
    
    //Start EM Iteration
    // printclusters(cls, K, raw_num_features);

    //start iteration
    int count = 0;
    while(count < termination_count) {
        //E-step = assign points to closest cluster
        rand_closestCentroid(cls, K, data, rows, num_data, raw_num_features);

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

// Data oblivious centroid recomputation for test data 
void rand_recomputeCurrCentroid(struc_cluster *cls, int curr, double *data, bool row, uint32_t num_raw_features) {

    // compute class proportions
    // use only real data for computation
    // ignore fake data in an oblivious manner 

    // recompute centrold 
    int decoy_if;
    if(row) {
        decoy_if = 1;
    } else {
        decoy_if = 0;
    }

    for(int i=0; i<num_raw_features; i++) {
        cls[curr].clus_data_sum[i] += (data[i]*decoy_if);
    }
    cls[curr].data_len += decoy_if;
    cls[curr].empty = false;
    

    // recompute centrold 
    for(int j=0; j<num_raw_features; j++) {
        cls[curr].centroid[j] = (cls[curr].clus_data_sum[j] / cls[curr].data_len);    
    }
}


//Predict class label and output size of novel class buffer.
int rand_test(double *data, bool row, uint32_t num_features, struc_cluster *cls, uint32_t K) {
        
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
    int res = oequal_int(cls[min_cls].class_label, (int)data[num_features-1]);

    //recompute centroid of min_cls cluster 
    rand_recomputeCurrCentroid(cls, min_cls, data, row, num_features-1);

    return res;
}


/////////////////////////////////////////////////////////////////////////////////////////////////

struc_cluster *rclusters;

//ocall receiver
void startKMRandTraining(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t fake_size, uint32_t K_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    uint32_t num_values = 11;
    int num_iteration = (num_data/iter_size - 1); //num of chunks
    int FAKE_PROP = fake_size;
    int K = K_size;

    uint32_t  termination_count = 10; //EM

    //GET NEW DATA
    uint32_t start_data = 0;
    uint32_t datasize = iter_size; //200;

    uint32_t len = sizeof(char)*datasize*num_features*100;

    //incoming data buffer
    double **data = new double*[datasize*FAKE_PROP];
    for(int i=0; i<datasize*FAKE_PROP; i++) {
        data[i] = new double[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = -1;
        }
    }

    //MODEL PARAMETERS
    rclusters = new struc_cluster[num_classes*5];

    //initialize pointer to real data 
    bool *rows = new bool[datasize*FAKE_PROP];
    for(int i=0; i<datasize; i++) {
        rows[i] = true;
    }
    //initialize pointer to fake data 
    for(int i=datasize; i<datasize*FAKE_PROP; i++) {
        rows[i] = false;
    }

    //START STREAMING
    int iteration_count = 0;
    int correct = 0;
    int total_pred = 0;
    double acc = 0;

    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*100;
    char *s = new char[rlen];

    // printf("Reading data from app.\n");
    int row_count = 0;
    for(int i=start_data; i<(start_data+datasize); i++) {
        ocall_read_data(i, 1, s, rlen);
        // decrypt_data(s, rlen)
        extract_features(row_count, data, s, datasize, num_features);
        ++row_count;
        
    }
    printf("row_count = %d\n", row_count);
    printf("No. of features = %d\n" , num_features);

    add_decoy_data(data, datasize, FAKE_PROP, num_features, num_classes, rows);

    //INITIAL CLUSTER

    printf("Initializing rclusters\n");
    for(int i=0; i<K; i++) {
        //set centroid features, data, class proportions to -1
        rclusters[i].cluster_num = i;
        rclusters[i].centroid = new double[num_features-1];
        rclusters[i].clus_data_sum = new double[num_features-1];
        for(int j=0; j<num_features-1; j++) {
            rclusters[i].centroid[j] = -1;
            rclusters[i].clus_data_sum[j] = 0;
        }
        rclusters[i].data_len = 0;
        rclusters[i].class_prop = new int[num_classes];
        for(int j=0; j<num_classes; j++) {
            rclusters[i].class_prop[j] = 0;
        }
        rclusters[i].num_classes = num_classes;
        rclusters[i].class_label = -1;
        rclusters[i].max_data_dist = 0;
        rclusters[i].empty = true;
    }
    
    //TRAIN DECISION TREE;
    printf("Perform EM clustering.\n");
    rand_EM(data, rows, datasize*FAKE_PROP, num_features-1, K, rclusters, termination_count);
    printf("rclusters completed learned.\n");

    for(int i=0; i<datasize*FAKE_PROP; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] rows;
    delete [] s;
}



//ocall receiver
void startKMRandTesting(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t fake_size, uint32_t K_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    uint32_t num_values = 11;
    int num_iteration = (num_data/iter_size - 1); //num of chunks
    int FAKE_PROP = fake_size;
    int K = K_size;

    uint32_t  termination_count = 10; //EM

    //GET NEW DATA
    uint32_t start_data = 0;
    uint32_t datasize = iter_size; //200;

    uint32_t len = sizeof(char)*datasize*num_features*100;

    //incoming data buffer
    double **data = new double*[datasize*FAKE_PROP];
    for(int i=0; i<datasize*FAKE_PROP; i++) {
        data[i] = new double[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = -1;
        }
    }

    //MODEL PARAMETERS

    //initialize pointer to real data 
    bool *rows = new bool[datasize*FAKE_PROP];
    for(int i=0; i<datasize; i++) {
        rows[i] = true;
    }
    //initialize pointer to fake data 
    for(int i=datasize; i<datasize*FAKE_PROP; i++) {
        rows[i] = false;
    }

    //START STREAMING
    int iteration_count = 0;
    int correct = 0;
    int total_pred = 0;
    double acc = 0;

    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*10;
    char *s = new char[rlen];
    char *res_out = new char[datasize*FAKE_PROP];

    do {
        // printf("Reading data from app.\n");
        int row_count = 0;
        for(int i=start_data; i<(start_data+datasize); i++) {
            ocall_read_data(i, 1, s, rlen);
            // decrypt_data(s, rlen)
            extract_features(row_count, data, s, datasize, num_features);
            ++row_count;
            
        }
        printf("row_count = %d\n", row_count);
        printf("No. of features = %d\n" , num_features);

        add_decoy_data(data, datasize, FAKE_PROP, num_features, num_classes, rows);

        //INITIAL CLUSTER
        
        //TEST NEW DATA
        printf("Testing iteration ...\n");
        
        for(int i=0; i<datasize*FAKE_PROP; i++) {
            int res = rand_test(data[i], rows[i], num_features, rclusters, K);
            int decoy_if = oblivious_check(rows, i);
            if(res > 0) {
                correct += decoy_if;   
            }
            res_out[i] = (char)res;
            total_pred += decoy_if;
        }

        // encrypt_data(res_out, datasize*FAKE_PROP);

        if(total_pred > 0) {
            acc = (double) correct / total_pred;
            printf("Acc=%f\n", acc);
        }        

        //NEXT ITERATION
        start_data += datasize;
        for(int i=0; i<datasize*FAKE_PROP; i++) {
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

    delete [] res_out;
    
    
    //delete rclusters
    for(int i=0; i<K; i++) {
        delete [] rclusters[i].centroid;
        delete [] rclusters[i].clus_data_sum;
        delete [] rclusters[i].class_prop;
    }
    delete [] rclusters;

    for(int i=0; i<datasize*FAKE_PROP; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] rows;
    delete [] s;

    printf("END OF ENCLAVE STREAM MINING.\n");
    ocall_print_acc(acc);
}

