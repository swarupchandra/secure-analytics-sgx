

#include "Enclave.h"
#include "Enclave_t.h"

#include "analytics_util.h"
#include "decision_tree.h"


Node* DTLearn(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features) {
    Node *root = new Node;
    initializeNode(root);

    printf("Computing gain tree\n");
    computeGainTree(data, num_data, num_features, num_values, num_classes, rows, features, root);

    return root;
}


bool DTtest(int *data, int num_features, Node *node, int pred_class_label) {
    
    bool result = false;
    if(node->leaf && node->split_attr == -1) {
        pred_class_label = node->class_label;
    } 

    if(node->num_children > 0 && node->split_attr != -1) {
        int val = data[node->split_attr];
        result = DTtest(data, num_features, &node->children[val], pred_class_label);
    } else {
        if(data[num_features-1] == pred_class_label) {
            result = true;
        } else {
            result = false;
        }
    }
    
    return result;
}



// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%% MAIN PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Node *root = NULL;

//ocall receiver
void startDTTraining(int num_data, int num_features, int num_classes, int iter_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    int num_values = 11;
    int num_iteration = (num_data/iter_size - 1); //num of chunks

    //GET NEW DATA
    int start_data = 0;
    int datasize = iter_size;

    int len = sizeof(char)*datasize*num_features*10;
    char data_str[len];

    int **data = new int*[datasize];
    for(int i=0; i<datasize; i++) {
        data[i] = new int[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = 0;
        }
    }

    //MODEL PARAMETERS

    //set features and rows that are valid
    int *features = new int[num_features];
    for(int i=0; i<num_features; i++) {
        features[i] = i;
    }

    bool *rows = new bool[datasize];
    for(int i=0; i<datasize; i++) {
        rows[i] = true;
    }

    //START STREAMING
    int iteration_count = 0;
    //buffer to read single data:
    int rlen = sizeof(char)*1*num_features*10;
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
        
    //TRAIN DECISION TREE;
    printf("Learning decision Tree\n");
    root = DTLearn(data, datasize, num_features, num_values, num_classes, rows, features);
    printf("Decision tree learned.\n");
    
    // printDT(root, 0);
    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;

    delete [] features;
    delete [] rows;

}

//ocall receiver
void startDTTesting(int num_data, int num_features, int num_classes, int iter_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );
    
    int num_values = 11;
    int num_iteration = (num_data/iter_size - 1); //num of chunks

    //GET NEW DATA
    int start_data = 0;
    int datasize = iter_size;

    int len = sizeof(char)*datasize*num_features*10;
    char data_str[len];

    int **data = new int*[datasize];
    for(int i=0; i<datasize; i++) {
        data[i] = new int[num_features];
        for(int j=0; j<num_features; j++) {
            data[i][j] = 0;
        }
    }

    //MODEL PARAMETERS

    //set features and rows that are valid
    int *features = new int[num_features];
    for(int i=0; i<num_features; i++) {
        features[i] = i;
    }

    bool *rows = new bool[datasize];
    for(int i=0; i<datasize; i++) {
        rows[i] = true;
    }

    //START STREAMING
    int iteration_count = 0;
    //buffer to read single data:
    int rlen = sizeof(char)*1*num_features*10;
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
        
        //TEST NEW DATA and UPDATE CPD BUFFER
        printf("Testing ...\n");
        // int true_count = 0;
        char *res_out = new char[datasize];
        for(int i=0; i<datasize; i++) {
            bool result = DTtest(data[i], num_features, root, -1);
            if(result) { 
                ++correct; 
            }
            res_out[i] = (char)result;
            ++total_pred;
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
            if(root != NULL) {
                printf("END\n");
                deleteTree(root);
            }
            break;
        }
        
        ++iteration_count;
        printf("******************* NEXT ITERATION (%d, %d) ***************************\n", start_data, (start_data + datasize));

    } while (start_data < num_data);
    
    
    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;

    delete [] features;
    delete [] rows;

    printf("END OF ENCLAVE MINING.\n");
    ocall_print_acc(stream_acc);
}