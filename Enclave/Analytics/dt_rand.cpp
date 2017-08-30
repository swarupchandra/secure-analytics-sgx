
#include <stdarg.h>
#include <stdio.h>      /* vsnprintf */
#include <cstring>
#include <math.h>

#include <sgx_trts.h>
#include <vector>
#include <queue>
#include <string>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */
#include "sgx_tcrypto.h"

#include "analytics_util.h"
#include "decision_tree.h"
#include "rand_primitives.h"



/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
//////////////////// Data Randomization /////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////


// A data oblivious method to select (or not select) decoy data.
//If decoy data is to be selected (decoy=true), then select 50% of data instances.
int decoyCorrection(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features, int max_index, bool decoy) {

    // enabling rows to use fake data
    // this indicates that decoy data usage is new. Once fake, further function calls remain fake 
    // Get values of current active rows 
    bool *prev_active_rows = new bool[num_data];

    // printf("Memory to hold previous row values allocated.\n");
    for(int i=0; i<num_data; i++) {
        prev_active_rows[i] = rows[i];
        rows[i] = false;
    }

    // randomly select rows in "rows" to equal real_rows, if fake_rows = 0 - this indicates that decoy data usage is new 
    for(int i=0; i<num_data; i++) {
        int random = get_random_int(num_data);
        if(random < num_data/2) {
            rows[i] = true;
        } else {
            rows[i] = false;
        }
    }

    // correct row activation depending on decoy 
    for(int i=0; i<num_data; i++) {
       // if !correction, then rows[i] = prev_active_rows[i].
       // Equivalent boolean formula: (~correction&prev_active_rows[i]) | (correction&rows[i])
        rows[i] = ((~decoy & prev_active_rows[i]) | (decoy & rows[i]));
    }
    
    delete [] prev_active_rows;

    int random_max = get_random_int(num_features);
    int decoy_if = oequal_int(max_index, -1);
    max_index = (decoy_if*random_max) + ((1-decoy_if)*max_index);
    return max_index;
}


/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
//////////////////// Decision Tree //////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////


struct RNode {
    RNode *parent;
    RNode *children;
    int num_children;
    int parent_value;
    int split_attr;
    int class_label;
    bool leaf;
    bool decoy;
};

void initializeRNode(RNode *node, bool decoy) {
    node->parent = NULL;
    node->children = NULL;
    node->num_children = 0;
    node->parent_value = -1;
    node->split_attr = -1;
    node->class_label = -1;
    node->leaf = false;
    node->decoy = decoy;
}

// Forward declaration
void computeGainRTree(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features, RNode *t, int depth);

// Grow tree by creating children
void createRChildren(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features, RNode *t, int max_index, int depth, bool decoy) {
    t->num_children = num_values;
    t->children = new RNode[t->num_children];
    for(int i=0; i<t->num_children; i++) {
        initializeRNode(&t->children[i], decoy);
        t->children[i].parent = t;
        t->children[i].parent_value = i;
    }

    // printf("Tree initialized ...\n");

    bool *orig_rows = new bool[num_data];
    for(int i=0; i<num_data; i++) {
        orig_rows[i] = rows[i];
    }
    // printf("Original rows value saved ...\n");

    // decoy: MAKE DATA OBLIVIOUS
    //1. max_index can be -1. Use random value
    //2. rows indices to data matrix should use decoy data
    //3. change decoy labels
    max_index = decoyCorrection(data, num_data, num_features, num_values, num_classes, rows, features, max_index, decoy);
    // printf("Decoy corrected ...\n");

    //mark feature selected used.
    int orig_feature = features[max_index];
    features[max_index] = -1;
    
    //recursively call computeGainRTree for each child, adjust data accordingly.
    for(int i=0; i<t->num_children; i++) {
        //create new data - ignore rows with parent value for parent attribute, and remove parent attribute column
        int num_new_rows = 0;
        int num_new_cols = 0;

        //get number of columns for child
        for(int p=0; p<num_features-1; p++) {
            if(features[p] != -1) {
                num_new_cols += 1;
            }
        }

        if(num_new_cols < 1) {
            t->children[i].num_children = 0;
            t->children[i].split_attr = -1;
            t->children[i].class_label = -1;
            continue;
        }

        //get new rows for the child node
        bool *new_rows = new bool[num_data];
        for(int p=0; p<num_data; p++) {
            new_rows[p] = rows[p];
            if(rows[p]) {
                int decoy_row = oequal_int(data[p][max_index], t->children[i].parent_value);
                num_new_rows += decoy_row;
                if(data[p][max_index] == t->children[i].parent_value) {
                    new_rows[p] = true;
                } else {
                    new_rows[p] = false;
                }
            }
        }

        if(num_new_rows == 0) {
            // no data at child
            t->children[i].num_children = 0;
            t->children[i].split_attr = -1;
            t->children[i].class_label = -1;
            delete [] new_rows;
            continue;
        }

        //recursive call
        computeGainRTree(data, num_data, num_features, num_values, num_classes, new_rows, features, &t->children[i], depth);    
        delete [] new_rows;
    }

    //restore original feature
    features[max_index] = orig_feature;
	//restore original rows
    for(int i=0; i<num_data; i++) {
        rows[i] = orig_rows[i];
    }
    delete [] orig_rows;
}


//num_data = dataset size
//num_features = features in order with last feature as class.
//num_values = number of discrete values for each feature (starting from 0).
void computeGainRTree(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features, RNode *t, int depth) {

    if(depth == MAX_DEPTH) {
        return;
    } 

    int valid_rows=0, valid_cols = 0;
    for(int i=0; i<num_data; i++) {
        valid_rows += oblivious_check(rows, i);
    }

    for(int i=0; i<num_features; i++) {
        ++valid_cols;
    }
    
    int max_class_label = -1;

    // printf("Computing gain G ...\n");
    double prop[num_classes];
    for(int i=0; i<num_classes; i++) {
        prop[i] = 0;
    }
    for(int i=0; i<num_data; i++) {
        if(rows[i]) {
            int class_val = data[i][num_features-1];
            ++prop[class_val];
        }
    }

    bool complete = false;
    for(int i=0; i<num_classes; i++) {
        prop[i] /= valid_rows;
        if(prop[i] == 1) {
            max_class_label = i;
            complete = true;
        }
    }

    int max_feature = -1;
    int max_index = -1;

    // if complete, end legitimate tree. Check to add decoy tree
    if(complete) {
        t->class_label = max_class_label;
        t->split_attr = -1;
        t->num_children = 0;
        t->leaf = true;
        // printf("Zero gain and not split. Class label=%d\n", t->class_label);

        //if depth less than MAX_DEPTH, add dummy nodes using dummy data
        if(depth < MAX_DEPTH) {
            // create children using dummy data
            createRChildren(data, num_data, num_features, num_values, num_classes, rows, features, t, max_index, depth+1, true);
        }
        return;
    }

    int def_max_class = -1;
    double max_prop = 0;
    for(int i=0; i<num_classes; i++) {
        if(prop[i] > max_prop) {
            max_prop = prop[i];
            def_max_class = i;
        }
    }

    //compute Entropy(S)
    double S = computeEntropy(prop, num_classes);

    //Compute max gain 
    double max_gain = 0;
    
    //class vs feature
    int num_val_count[num_classes][num_values];

    // For each feature, compute gain
    for(int i=0; i<num_features-1; i++) {

        if(features[i] < 0) {
            continue;
        }

        // reset count 
        for(int p=0; p<num_classes; p++) {
            for(int q=0; q<num_values; q++) {
                num_val_count[p][q] = 0;
            }
        }
        
        for(int j=0; j<num_data; j++) {
            if(rows[j]) {
                int fea_val = data[j][i];
                int class_val = data[j][num_features-1];
                ++num_val_count[class_val][fea_val];
            }
        }
        
        double S_fea = 0;

        for(int j=0; j<num_values; j++) {
            for(int p=0; p<num_classes; p++) {
                prop[p] = 0;
            }

            double fea_total = 0;
            for(int p=0; p<num_classes; p++) {
                prop[p] = num_val_count[p][j];
                fea_total += num_val_count[p][j];
            }

            if(fea_total == 0) {
                continue;
            }

            for(int p=0; p<num_classes; p++) {
                prop[p] /= fea_total;
            }
            
            double sub_entropy = computeEntropy(prop, num_classes);
            S_fea += (fea_total * sub_entropy / valid_rows);
        }
        
        double gain = S - S_fea;
        
        //TODO oblivious max implementation
        if(gain > max_gain) {
            max_gain = gain;
            max_feature = features[i];
            max_index = i;
        }
    }
    
    //set tree RNode with max gain feature
    t->split_attr = max_feature;
    t->num_children = num_values;
    t->class_label = max_class_label;
    t->leaf = false;

    // if max gain is 0, return
    if(max_gain == 0) {
        t->split_attr = -1;
        t->num_children = 0;

        if(max_class_label == -1) {
            t->class_label = def_max_class;
            t->leaf = true;
        }

        //if depth less than MAX_DEPTH, add dummy nodes using dummy data
        if(depth < MAX_DEPTH) {
            // create children using dummy data
            createRChildren(data, num_data, num_features, num_values, num_classes, rows, features, t, max_index, depth+1, true);
        }

        return;
    }
    
    createRChildren(data, num_data, num_features, num_values, num_classes, rows, features, t, max_index, depth+1, t->decoy);
}


// Update: Learning need not be data oblivious.
RNode* DTLearn_rand(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features) {
    RNode *root_r = new RNode;
    initializeRNode(root_r, false);

    printf("Computing gain tree\n");
    computeGainRTree(data, num_data, num_features, num_values, num_classes, rows, features, root_r, 0);
    printf("Gain tree obtained.\n");

    return root_r;
}

// Data oblivious testing / evaluation
bool DTtest_rand(int *data, int num_features, RNode *node, int pred_class_label, bool result) {
    if(node->leaf && !node->decoy) {
        pred_class_label = node->class_label;
        if(data[num_features-1] == pred_class_label) {
            result = true;
        } else {
            result = false;
        }
    } 

    if(node->num_children > 0 && node->split_attr != -1) {
        int val = data[node->split_attr];
        result = DTtest_rand(data, num_features, &node->children[val], pred_class_label, result);
    }
    return result;
}

void deleteRTree(RNode *n) {
    if(n->num_children > 0) {
        for(int i=0; i<n->num_children; i++) {
            if(n->children[i].num_children > 0) {
                deleteRTree(&n->children[i]);
            }
        }
        delete [] n->children;
    }
    // delete n;
}


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%% MAIN PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


RNode *root_r = NULL;

//ocall training
void startDTRandTraining(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t fake_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    int num_values = 11;
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
    
    //set features and rows that are valid
    int *features = new int[num_features];
    for(int i=0; i<num_features; i++) {
        features[i] = i;
    }

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

    int correct = 0;
    int total_pred = 0;
    double stream_acc = 0;

    printf("Reading data from app.\n");
    int row_count = 0;
    for(int i=start_data; i<(start_data+datasize); i++) {
        ocall_read_data(i, 1, s, rlen);
        // decrypt_data(s, rlen)
        extract_features(row_count, data, s, datasize, num_features);
        ++row_count;        
    }
    printf("row_count: %d\n", row_count);

    //CREATE DUMMY DATA
    printf("Creating dummy data and randomizing.\n");
    add_decoy_data(data, datasize, FAKE_PROP, num_features, num_classes, rows);
    printf("Data randomized.\n");
        
    //TRAIN DECISION TREE;
    printf("Learning decision Tree\n");
    root_r = DTLearn_rand(data, datasize*FAKE_PROP, num_features, num_values, num_classes, rows, features);
    printf("Decision tree learned.\n");

    
    for(int i=0; i<datasize*FAKE_PROP; i++) {
        delete [] data[i];
    }
    delete [] data;

    delete [] features;
    delete [] rows;
    delete [] s;
}


//ocall testing
void startDTRandTesting(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t fake_size) {

    //PUBLIC PARAMETERS
    printf("num_data=%d, num_features=%d, and num_classes=%d\n", num_data, num_features, num_classes );

    int num_values = 11;
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
    
    //set features and rows that are valid
    int *features = new int[num_features];
    for(int i=0; i<num_features; i++) {
        features[i] = i;
    }

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
    // int max_level = 0;

    int correct = 0;
    int total_pred = 0;
    double stream_acc = 0;

    if(root_r == NULL) {
        printf("root_r NOT INITIALIZED");
        return;
    }

    char *res_out = new char[datasize*FAKE_PROP];

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

        //CREATE DUMMY DATA
        add_decoy_data(data, datasize, FAKE_PROP, num_features, num_classes, rows);
        // printf("Data randomized.\n");

        //TEST NEW DATA and UPDATE CPD BUFFER
        printf("Testing ...\n");
        // int true_count = 0;
        
        for(int i=0; i<datasize*FAKE_PROP; i++) {
            bool result = DTtest_rand(data[i], num_features, root_r, -1, false);

            if(result) { 
                correct += oblivious_check(rows, i);
            }
            total_pred += oblivious_check(rows, i);
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
            if(root_r != NULL) {
                printf("END\n");
                deleteRTree(root_r);
            }
            break;
        }
        
        ++iteration_count;
        printf("******************* NEXT ITERATION (%d, %d) ***************************\n", start_data, (start_data + datasize));

    } while (start_data < num_data);
    
    
    for(int i=0; i<datasize*FAKE_PROP; i++) {
        delete [] data[i];
    }
    delete [] data;

    delete [] features;
    delete [] rows;
    delete [] s;
    delete [] res_out;

    printf("END OF ENCLAVE MINING.\n");
    ocall_print_acc(stream_acc);
}
