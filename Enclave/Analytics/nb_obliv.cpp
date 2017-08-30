#include <stdio.h>      /* vsnprintf */
#include <cstring>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */
#include "sgx_tcrypto.h"

#include "analytics_util.h"
#include "nbayes.h"


bool obliv_test_nb(int *data, NB *nb_oroot, int num_features, int num_classes, int num_values) {
    double max_prod = 0;
    int max_class = 0;
    //compute P(x|c) = argmax (P(c)*P(c|x1)*P(c|x2)*...)
    for(int i=0; i<num_classes; i++) {
        int p_c = nb_oroot->prob[i];
        for(int j=0; j<num_features; j++) {
            
            for(int k=0; k<num_values; k++) {
                int decoy_prob_if = oequal_int(k, data[j]);
                p_c *= ((nb_oroot->children[i].children[j].prob[k]*decoy_prob_if) + (1-decoy_prob_if));
            }
        }

        // decoy
        int decoy_max = ogreater_int(p_c, max_prod);
        max_prod = (p_c*decoy_max) + (max_prod*(1-decoy_max));
        max_class = (i*decoy_max) + (max_class*(1-decoy_max));
    }

    bool decoy_res;
    if(max_class != data[num_features-1]) {
        decoy_res = false;
    } else {
        decoy_res = true;
    }
    return decoy_res;
}


/////////////////////////////////////////////////////////

NB *nb_oroot = NULL;

//ocall training
void startNBOblivTraining(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size) {

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

    bool *rows = new bool[datasize];
    for(int i=0; i<datasize; i++) {
        rows[i] = true;
    }
    //initialize fake data
    for(int i=datasize; i<datasize; i++) {
        rows[i] = false;
    }

    //START STREAMING
    int iteration_count = 0;
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

    randomize_data_index(data, datasize, num_features);

    //TRAIN Naive Bayes mpdel;
    printf("Learning Naive Bayes\n");
    nb_oroot = obliv_learn_nb(data, rows, datasize, num_features, num_classes, num_values);
    printf("Naive Bayes learned.\n");

    // print_nb_tree(nb_oroot, num_features, num_classes, num_values);

    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] rows;

}

//ocall testing
void startNBOblivTesting(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size) {

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

    bool *rows = new bool[datasize];
    for(int i=0; i<datasize; i++) {
        rows[i] = true;
    }
    //initialize fake data
    for(int i=datasize; i<datasize; i++) {
        rows[i] = false;
    }

    //START STREAMING
    int iteration_count = 0;
    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*10;
    char *s = new char[rlen];

    int correct = 0;
    int total_pred = 0;
    double stream_acc = 0;

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

        randomize_data_index(data, datasize, num_features);
        
        //TEST NEW DATA and UPDATE CPD BUFFER
        printf("Testing ...\n");
        // int true_count = 0;
        char *res_out = new char[datasize];
        for(int i=0; i<datasize; i++) {
            bool result = obliv_test_nb(data[i], nb_oroot, num_features, num_classes, num_values);
            int decoy_if = oblivious_check(rows, i);
            if(result) {
                correct += decoy_if;
            }
            res_out[i] = (char)result;
            total_pred += decoy_if;
        }

        // encrypt_data(res_out, datasize);
        delete [] res_out;

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
            if(nb_oroot != NULL) {
                printf("END\n");
                deleteNB(nb_oroot, num_features, num_classes);
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
    delete [] rows;

    printf("END OF ENCLAVE STREAM TESTING.\n");
    ocall_print_acc(stream_acc);
}
