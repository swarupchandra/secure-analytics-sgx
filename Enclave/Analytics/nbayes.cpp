#include <stdio.h>      /* vsnprintf */
#include <cstring>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */

#include "nbayes.h"
#include "analytics_util.h"


NB* initialize_NB(int num_classes, int num_features, int num_values) {
    NB *root = new NB();
    root->label = -1;
    root->num_children = num_classes;
    root->prob = new double[num_classes]; //prob(class)
    root->values = new int[num_classes];
    root->children = new NB[num_classes];
    for(int i=0; i<num_classes; i++) {
        // level = class labels
        root->values[i] = 0; // count class for prob(class)
        root->prob[i] = 0;
        root->children[i].label = i;
        root->children[i].prob = NULL; // not valid
        root->children[i].num_children = num_features;
        root->children[i].values = NULL; // not valid
        root->children[i].children = new NB[num_features];
        for(int j=0; j<num_features; j++) {
            // level = feature labels
            NB *feature_child = &root->children[i].children[j];
            feature_child->label = j;
            feature_child->prob = new double[num_values]; // prob(f=a|class)
            feature_child->num_children = num_values;
            feature_child->values = new int[num_values];
            feature_child->children = NULL; // not valid
            for(int k=0; k<num_values; k++) {
                // level = values of features
                feature_child->values[k] = 0; // count f=a for prob(f=a|class)
                feature_child->prob[k] = 0;
            }
        }
    }
    return root;
}


void print_nb_tree(NB *root, int num_features, int num_classes, int num_values) {
    printf("label=%d\nprob=", root->label);
    for(int i=0; i<num_classes; i++) {
        printf("%f, ", root->prob[i]);
    }
    printf("\n");
}


void deleteNB(NB *root, int num_features, int num_classes) {
    for(int i=0; i<num_classes; i++) {
        delete [] root->children[i].prob;
        delete [] root->children[i].values;
        for(int j=0; j<num_features; j++) {
            delete [] root->children[i].children[j].prob;
            delete [] root->children[i].children[j].values;
        }
        delete [] root->children[i].children;
    }
    delete [] root->values;
    delete [] root->prob;
    delete [] root->children;
    delete root;
}


/////////////////////////////////////////////////////////

bool nb_test(int *data, NB *root, int num_features, int num_classes) {
    double max_prod = 0;
    int max_class = 0;
    //compute P(x|c) = argmax (P(c)*P(c|x1)*P(c|x2)*...)
    for(int i=0; i<num_classes; i++) {
        int p_c = root->prob[i];
        for(int j=0; j<num_features; j++) {
            p_c *= root->children[i].children[j].prob[data[j]];
        }
        if(p_c > max_prod) {
            max_prod = p_c;
            max_class = i;
        }
    }

    if(max_class != data[num_features-1]) {
        return false;
    } else {
        return true;
    }
}


NB* obliv_learn_nb(int **data, bool *rows, int num_data, int num_features, int num_classes, int num_values) {
    //initialize NB tree
    NB *root = initialize_NB(num_classes, num_features, num_values);
    printf("Initialized NB structure.\n");

    // for each data instance, update count in the NB tree
    // in an oblivious manner
    for(int i=0; i<num_data; i++) {
        int decoy_if = oblivious_check(rows, i);
        int label = data[i][num_features-1];
        root->values[label] += decoy_if;
        NB *fea_node = &root->children[label];
        for(int j=0; j<num_features-1; j++) {
            int val = data[i][j];
            fea_node->children[j].values[val] += decoy_if;
        }
    }

    // compute probability
    for(int i=0; i<num_classes; i++) {
        root->prob[i] = (double) root->values[i] / num_data;
        int total_class_count = root->values[i];
        int decoy_total_class = oequal_int(total_class_count, 0);
        total_class_count += decoy_total_class;
        for(int j=0; j<num_features; j++) {
            for(int k=0; k<num_values; k++) {
                root->children[i].children[j].prob[k] = (double) root->children[i].children[j].values[k] / total_class_count;
            }
        }
    }

    return root;
}