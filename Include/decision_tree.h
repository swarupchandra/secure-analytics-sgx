
struct Node {
    Node *parent;
    Node *children;
    int num_children;
    int parent_value;
    int split_attr;
    int class_label;
    bool leaf;
};

void initializeNode(Node *node);
double computeEntropy(double *prop, int num_classes);

void computeGainTree(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features, Node *t);
void createChildren(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features, Node *t, int max_index);
void computeGainTree(int **data, int num_data, int num_features, int num_values, int num_classes, bool *rows, int *features, Node *t);
void printDT(Node *node,  int count);
void deleteTree(Node *n);