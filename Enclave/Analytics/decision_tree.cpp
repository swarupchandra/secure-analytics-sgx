#include <math.h>

#include "decision_tree.h"
#include "Enclave.h"
#include "Enclave_t.h"

void initializeNode(Node *node) {
    node->parent = NULL;
    node->children = NULL;
    node->num_children = 0;
    node->parent_value = -1;
    node->split_attr = -1;
    node->class_label = -1;
    node->leaf = false;
}

// Compute Entropy of given class proportions
double computeEntropy(double *prop, int num_classes) {
    double entropy = 0;
    for(int i=0; i<num_classes; i++) {
        if(prop[i] > 0) {
            entropy += (prop[i] * log(prop[i]));
        }
    }
    return -1*entropy;
}

// Grow tree by creating children
void createChildren(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features, Node *t, int max_index) {
	t->num_children = num_values;
    t->children = new Node[t->num_children];
    for(int i=0; i<t->num_children; i++) {
        initializeNode(&t->children[i]);
        t->children[i].parent = t;
        t->children[i].parent_value = i;
    }

    //mark feature selected used.
    int orig_feature = features[max_index];
    features[max_index] = -1;
    
    //recursively call computeGainTree for each child, adjust data accordingly.
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
                if(data[p][max_index] == t->children[i].parent_value) {
                    num_new_rows += 1;
                    new_rows[p] = true;
                } else {
                    new_rows[p] = false;
                }
            }
        }

        if(num_new_rows == 0) {
            // printf("no data at child %d\n", i);
            t->children[i].num_children = 0;
            t->children[i].split_attr = -1;
            t->children[i].class_label = -1;
            delete [] new_rows;
            continue;
        }

        //recursive call
        computeGainTree(data, num_data, num_features, num_values, num_classes, new_rows, features, &t->children[i]);

        delete [] new_rows;
    }

    //restore original feature
    features[max_index] = orig_feature;
}


//num_data = dataset size
//num_features = features in order with last feature as class.
//num_values = number of discrete values for each feature (starting from 0).
void computeGainTree(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features, Node *t) {

    uint32_t valid_rows=0, valid_cols = 0;
    for(int i=0; i<num_data; i++) {
        if(rows[i]) {
            ++valid_rows;
        }
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

    //count for class vs feature value 
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
        
        // compute distribution by class label vs value count 
        for(int j=0; j<num_data; j++) {
            if(rows[j]) {
                int fea_val = data[j][i];
                int class_val = data[j][num_features-1];
                ++num_val_count[class_val][fea_val];
            }
        }

        // Gain : S - sigma(S_fea) for all fea in values 
        // S_fea = p(fea)*entropy(fea)
        double S_fea = 0;

        //For each value, compute entropy and gain 
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
    
    //set tree node with max gain feature
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

        return;
    }

    createChildren(data, num_data, num_features, num_values, num_classes, rows, features, t, max_index);
}

void printDT(Node *node,  int count) {

    for(int i=0; i<count; i++) {
        printf("*");
    }
    printf(" Value:%d", node->parent_value);
    printf(" Split:%d", node->split_attr);
    printf(" Class:%d", node->class_label);
    if(node->leaf) {
        printf("LEAF");
    }
    
    printf("\n");

    for(int i=0; i<node->num_children; i++) {
        printDT(&node->children[i], count+1);
    }
}

void deleteTree(Node *n) {
    if(n->num_children > 0) {
        for(int i=0; i<n->num_children; i++) {
            if(n->children[i].num_children > 0) {
                deleteTree(&n->children[i]);
            }
        }
        delete [] n->children;
    }
    // delete n;
}