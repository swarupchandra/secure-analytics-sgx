#include <stdio.h>      /* vsnprintf */
#include <cstring>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */
#include "sgx_tcrypto.h"

#include "analytics_util.h"
#include "nbayes.h"
#include "rand_primitives.h"



///////////////////////////////////////////////////////////////////////////////


NB *nb_rroot = NULL;

//ocall training
void startNBRandTraining(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t fake_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );
    
    int num_values = 1000;
    int FAKE_PROP = fake_size;
    int num_iteration = (num_data/iter_size - 1); //num of chunks

    //GET NEW DATA
    int start_data = 0;
    int datasize = iter_size;

    int len = sizeof(char)*datasize*num_features*10;
    char data_str[len];

    int **data = new int*[datasize*FAKE_PROP];
    for(int i=0; i<datasize*FAKE_PROP; i++) {
        data[i] = new int[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = 0;
        }
    }

    //MODEL PARAMETERS

    bool *rows = new bool[datasize*FAKE_PROP];
    for(int i=0; i<datasize; i++) {
        rows[i] = true;
    }
    //initialize fake data
    for(int i=datasize; i<datasize*FAKE_PROP; i++) {
        rows[i] = false;
    }

    //START STREAMING
    int iteration_count = 0;
    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*100;
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

    //CREATE DUMMY DATA
    // printf("Creating dummy data and randomizing.\n");
    add_decoy_data(data, datasize, FAKE_PROP, num_features, num_classes, rows);

    //TRAIN Naive Bayes mpdel;
    printf("Learning Naive Bayes\n");
    nb_rroot = obliv_learn_nb(data, rows, datasize*FAKE_PROP, num_features, num_classes, num_values);
    printf("Naive Bayes learned.\n");

    // print_nb_tree(nb_rroot, num_features, num_classes, num_values);
    for(int i=0; i<datasize*FAKE_PROP; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] rows;

}

//ocall testing
void startNBRandTesting(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t fake_size) {
    
    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );
    
    int num_values = 1000;
    int FAKE_PROP = fake_size;
    int num_iteration = (num_data/iter_size - 1); //num of chunks

    //GET NEW DATA
    int start_data = 0;
    int datasize = iter_size;

    int len = sizeof(char)*datasize*num_features*100;
    char data_str[len];

    int **data = new int*[datasize*FAKE_PROP];
    for(int i=0; i<datasize*FAKE_PROP; i++) {
        data[i] = new int[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = 0;
        }
    }

    //MODEL PARAMETERS

    bool *rows = new bool[datasize*FAKE_PROP];
    for(int i=0; i<datasize; i++) {
        rows[i] = true;
    }
    //initialize fake data
    for(int i=datasize; i<datasize*FAKE_PROP; i++) {
        rows[i] = false;
    }

    //START STREAMING
    int iteration_count = 0;
    //buffer to read single data:
    uint32_t rlen = sizeof(char)*1*num_features*10;
    char *s = new char[rlen];
    char *res_out = new char[datasize*FAKE_PROP];

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

        //CREATE DUMMY DATA
        // printf("Creating dummy data and randomizing.\n");
        add_decoy_data(data, datasize, FAKE_PROP, num_features, num_classes, rows);

       // INITIAL
        
        //TEST NEW DATA and UPDATE CPD BUFFER
        printf("Testing ...\n");
        
        for(int i=0; i<datasize*FAKE_PROP; i++) {
            bool result = nb_test(data[i], nb_rroot, num_features, num_classes);
            int decoy_if = oblivious_check(rows, i);
            if(result) {
                correct += decoy_if;
            }
            total_pred += decoy_if;
            res_out[i] = (char)result;
        }

        // encrypt_data(res_out, datasize*FAKE_PROP);
        
        stream_acc = (double) correct / total_pred;
        printf("Acc=%f\n", stream_acc);

        //NEXT ITERATION
        start_data += datasize;
        for(int i=0; i<datasize*FAKE_PROP; i++) {
            for(int j=0; j<num_features; j++) {
                data[i][j] = -1;
            }
        }

        if(iteration_count == num_iteration) {
            if(nb_rroot != NULL) {
                printf("END\n");
                deleteNB(nb_rroot, num_features, num_classes);
            }
            break;
        }

        ++iteration_count;
        printf("******************* NEXT ITERATION (%d, %d) ***************************\n",start_data,(start_data + datasize));

    } while (start_data < num_data);


    for(int i=0; i<datasize*FAKE_PROP; i++) {
        delete [] data[i];
    }
    delete [] data;
    delete [] rows;
    delete [] res_out;

    printf("END OF ENCLAVE STREAM MINING.\n");
    ocall_print_acc(stream_acc);
}

