#include <stdarg.h>
#include <stdio.h>      /* vsnprintf */
#include <cstring>
#include <string>

#include <sgx_trts.h>
#include "sgx_tcrypto.h"

#include "Enclave.h"
#include "Enclave_t.h"

#include "analytics_util.h"
#include "decision_tree.h"

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
//////////////////// Decision Tree //////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////



int getTreeDepth(Node *n, int depth) {
    int max_depth = depth;
    if(n->num_children > 0) {
        for(int i=0; i<n->num_children; i++) {
            int curr_depth = getTreeDepth(&n->children[i], depth+1);
            if(curr_depth > max_depth) {
                max_depth = curr_depth;
            }
        }
    }
    return max_depth;
}



void makeTreeOblivious(Node *n, int depth, int num_values) {
    if(n->num_children > 0) {
        for(int i=0; i<n->num_children; i++) {
            makeTreeOblivious(&n->children[i], depth+1, num_values);
        }
    } else {
        if(depth < MAX_DEPTH) {
            // If leaf, get class value;
            n->leaf = false;
            n->num_children = num_values;
            int newsplit = n->num_children-1;
            n->split_attr = newsplit;
            n->children = new Node[n->num_children];
            for(int i=0; i<n->num_children; i++) {
                initializeNode(&n->children[i]);
                n->children[i].parent = n;
                n->children[i].parent_value = i;
                //decoy node at newsplit
                n->children[i].class_label = n->class_label;
                makeTreeOblivious(&n->children[i], depth+1, num_values);
            }
            
            
        } else {
            if(n->class_label > -1) {
                n->leaf = true;
            }
        }
    }
}


Node* DTLearn_obliv(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features) {
    Node *root_o = new Node;
    initializeNode(root_o);

    printf("Computing gain tree\n");
    computeGainTree(data, num_data, num_features, num_values, num_classes, rows, features, root_o);

    //compute max depth 
    int max_depth = getTreeDepth(root_o, 0);
    if(max_depth > MAX_DEPTH) {
        printf("Greater than max depth !!\n");
    }

    //make tree balanced
    makeTreeOblivious(root_o, 0, num_values);

//    printDT(root_o, 0);

    return root_o;
}


int DTtest_obliv(int *data, int num_features, Node *node, int pred_class_label) {
    // printf("%d ", node->split_attr);
    
    int result = 0;
    int decoy_label_if;
    if(node->leaf) {
        if(node->split_attr == -1) {
            decoy_label_if = 1;
        } else {
            //decoy
            decoy_label_if = 0;
        } 
    } else {
        //decoy
        if(node->split_attr == -1) {
            decoy_label_if = 0;
        } else {
            decoy_label_if = 0;
        } 
    }
    pred_class_label = (decoy_label_if*node->class_label) + ((1-decoy_label_if)*pred_class_label);

    if(node->num_children > 0) {
        int val = data[node->split_attr];
        
        int decoy_val_if = ogreater_int(node->class_label, -1);
        // if(node->class_label > -1) {
        //     decoy_val_if = 1;
        // } else {
        //     decoy_val_if = 0;
        // }
        val = (decoy_val_if*node->num_children-1) + ((1-decoy_val_if)*val);
        
        for(int i=0; i<node->num_children; i++) {
            int decoy_result = DTtest_obliv(data, num_features, &node->children[i], pred_class_label);

            int decoy_node_if = oequal_int(i, val);
            // if(i == val) {
            //     decoy_node_if = 1;
            // } else {
            //     decoy_node_if = 0;
            // }
            result = (decoy_node_if*decoy_result) + ((1-decoy_node_if)*result);
            
        }
    } else {
        result = oequal_int(data[num_features-1], pred_class_label);
        // if(data[num_features-1] == pred_class_label) {
        //     result = 1;
        // } else {
        //     result = 0;
        // }

    }
    
    return result;
}




// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%% MAIN PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Node *root_o = NULL;

void startDTOblivTraining(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size) {

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
    printf("row_count: %d\n", row_count);

    randomize_data_index(data, datasize, num_features);
        
    //TRAIN DECISION TREE;
    printf("Learning decision Tree\n");
    root_o = DTLearn_obliv(data, datasize, num_features, num_values, num_classes, rows, features);
    printf("Decision tree learned.\n");

    for(int i=0; i<datasize; i++) {
        delete [] data[i];
    }
    delete [] data;

    delete [] features;
    delete [] rows;
    delete [] s;
        
}

//ocall receiver
void startDTOblivTesting(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size) {
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
    double acc = 0;

    if(root_o == NULL) {
        printf("root_o NOT INITIALIZED");
        return;
    }

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
        printf("row_count: %d\n", row_count);

        randomize_data_index(data, datasize, num_features);

        //TEST NEW DATA and UPDATE CPD BUFFER
        printf("Testing ...\n");
        
        for(int i=0; i<datasize; i++) {
            int result = DTtest_obliv(data[i], num_features, root_o, -1);
            correct += result;
            res_out[i] = (char)result;
            ++total_pred;
        }
        
        // encrypt_data(res_out, datasize);
        
        acc = (double) correct / total_pred;
        printf("Acc=%f\n", acc);

        //NEXT ITERATION
        start_data += datasize;
        for(int i=0; i<datasize; i++) {
            for(int j=0; j<num_features; j++) {
                data[i][j] = -1;
            }
        }

        if(iteration_count == num_iteration) {
            if(root_o != NULL) {
                printf("END\n");
                deleteTree(root_o);
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
    delete [] s;
    delete [] res_out;

    printf("END OF ENCLAVE STREAM MINING.\n");
    ocall_print_acc(acc);
}
