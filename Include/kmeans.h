struct struc_cluster {
    uint32_t cluster_num;
    double *centroid;
    // int *data_index; //index of data of cluster
    int data_len; //data size
    int *class_prop; //class proportions
    int num_classes; //class array size
    int class_label; //class label with largest proportion
    uint32_t max_data_dist; //cluster radius
    bool empty;
    double *clus_data_sum; //saving m latest data points
};


void printClusters(struc_cluster *cls, uint32_t K, uint32_t raw_num_features);
double euclidean_distance(double *p1, double *p2, uint32_t num_raw_features);

void recomputeCentroid(struc_cluster *cls, uint32_t K, double **data, uint32_t num_raw_features);
void reinitializeCluster(struc_cluster *cls, uint32_t K);