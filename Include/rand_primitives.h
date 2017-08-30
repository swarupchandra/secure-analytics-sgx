

void generate_new_data(int **data, bool *rows, int num_data, int num_features);
void generate_new_data(double **data, bool *rows, int num_data, int num_features);
void add_decoy_data(int **data, int num_data, int fake_times, int num_features, int num_classes, bool *rows);
void add_decoy_data(double **data, int num_data, int fake_times, int num_features, int num_classes, bool *rows);