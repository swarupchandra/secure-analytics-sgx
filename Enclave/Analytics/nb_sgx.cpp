#include <stdio.h>      /* vsnprintf */
#include <cstring>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */
#include "sgx_tcrypto.h"

#include "analytics_util.h"
#include "nbayes.h"


NB* learn_naive_bayes(int **data, int num_data, int num_features, int num_classes, int num_values) {
    //initialize NB tree
    NB *nb_root = initialize_NB(num_classes, num_features, num_values);
    printf("Initialized NB structure.\n");

    // for each data instance, update count in the NB tree
    for(int i=0; i<num_data; i++) {
        int label = data[i][num_features-1];
        nb_root->values[label] += 1;
        NB *fea_node = &nb_root->children[label];
        for(int j=0; j<num_features-1; j++) {
            int val = data[i][j];
            fea_node->children[j].values[val] += 1;
        }
    }

    // compute probability
    for(int i=0; i<num_classes; i++) {
        nb_root->prob[i] = (double) nb_root->values[i] / num_data;
        int total_class_count = nb_root->values[i];
        if(total_class_count == 0) {
            continue;
        }
        for(int j=0; j<num_features; j++) {
            for(int k=0; k<num_values; k++) {
                nb_root->children[i].children[j].prob[k] = (double) nb_root->children[i].children[j].values[k] / total_class_count;
            }
        }
    }

    return nb_root;
}


//////////////////////////////////////////////////////////////////////////////


//MODEL
NB *nb_root = NULL;

//ocall training
void startNBTraining(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size) {
    
    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    int num_values = 1000;
    int num_iteration = (num_data/iter_size - 1); //num of chunks

    //GET NEW DATA
    int start_data = 0;
    int datasize = iter_size;

    int len = sizeof(char)*datasize*num_features*100;
    char data_str[len];

    int **data = new int*[datasize];
    for(int i=0; i<datasize; i++) {
        data[i] = new int[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = 0;
        }
    }

    //MODEL PARAMETERS

    //START STREAMING
    int iteration_count = 0;
    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*10;
    char *s = new char[rlen];
    // int max_level = 0;

    printf("Reading data from app.\n");
    int row_count = 0;
    for(int i=start_data; i<(start_data+datasize); i++) {
        ocall_read_data(i, 1, s, rlen);
        // decrypt_data(s, rlen)
        extract_features(row_count, data, s, datasize, num_features);
        ++row_count;
    }
    printf("row_count = %d\n", row_count);

    //TRAIN Naive Bayes mpdel;
    printf("Learning Naive Bayes\n");
    nb_root = learn_naive_bayes(data, datasize, num_features, num_classes, num_values);
    printf("Naive Bayes learned.\n");

    // print_nb_tree(nb_root, num_features, num_classes, num_values);
    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;

}


//ocall receiver
void startNBTesting(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size) {
    
    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    int num_values = 1000;
    int num_iteration = (num_data/iter_size - 1); //num of chunks

    //GET NEW DATA
    int start_data = 0;
    int datasize = iter_size;

    int len = sizeof(char)*datasize*num_features*100;
    char data_str[len];

    int **data = new int*[datasize];
    for(int i=0; i<datasize; i++) {
        data[i] = new int[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = 0;
        }
    }

    //MODEL PARAMETERS

    //START STREAMING
    int iteration_count = 0;
    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*10;
    char *s = new char[rlen];

    int correct = 0;
    int total_pred = 0;
    double stream_acc = 0;
    
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
        
        //TEST NEW DATA and UPDATE CPD BUFFER
        printf("Testing ...\n");        
        for(int i=0; i<datasize; i++) {
            bool result = nb_test(data[i], nb_root, num_features, num_classes);
            if(result) {
                ++correct;
            }
            res_out[i] = (char)result;
            ++total_pred;
        }

        // encrypt_data(res_out, datasize);
        
        stream_acc = (double) correct / total_pred;
        printf("Acc=%f\n", stream_acc);

        //NEXT ITERATION
        start_data += datasize;
        for(int i=0; i<datasize; i++) {
            for(int j=0; j<num_features; j++) {
                data[i][j] = -1;
            }
        }

        if(iteration_count == num_iteration) {
            if(nb_root != NULL) {
                printf("END\n");
                deleteNB(nb_root, num_features, num_classes);
            }
            break;
        }

        ++iteration_count;
        printf("******************* NEXT ITERATION (%d, %d) ***************************\n",start_data,(start_data + datasize));

    } while (start_data < num_data);


    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] res_out;

    printf("END OF ENCLAVE STREAM MINING.\n");
    ocall_print_acc(stream_acc);
}

