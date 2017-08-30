


struct NB {
    int label; // class label or feature label
    int *values; // count of children labels
    NB *children; // class or feature or feature value
    int num_children; //size of children
    double *prob; // probability or conditional probability
};

NB* initialize_NB(int num_classes, int num_features, int num_values);
void print_nb_tree(NB *root, int num_features, int num_classes, int num_values);
void deleteNB(NB *root, int num_features, int num_classes);

bool nb_test(int *data, NB *root, int num_features, int num_classes);
NB* obliv_learn_nb(int **data, bool *rows, int num_data, int num_features, int num_classes, int num_values);