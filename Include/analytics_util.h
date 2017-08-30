#include <sgx_trts.h>
#include "sgx_tcrypto.h"

#ifndef ANALYTICS_UTIL_H
#define ANALYTICS_UTIL_H

const int MAX_DEPTH = 5;

void nonobliv_sort(int **data, int num_features, int x[], int lo, int n);
void obliv_oddeven_sort(int **data, int num_features, int x[], int n);
void obliv_oddeven_sort(double **data, int num_features, int x[], int n);

void get_random_shuffle(int *list, int k);
void randomize_data_index(int **data, int num_data, int num_features);
void randomize_rand_index(int **data, int num_data, int num_features);
void randomize_rand_index(double **data, int num_data, int num_features);
int get_random_int(int k);
double get_random_double(double max, double min);

void encrypt_data(uint8_t data[], size_t length);
void decrypt_data(uint8_t encrypted_buffer[], size_t length);

void extract_features(int row_count, int **data, char *data_str, int datasize, int num_features);
void extract_features(int row_count, double **data, char *data_str, int datasize, int num_features);

int oblivious_check(bool *rows, int index);
int omax_int(int x, int y );
double omax_double(double x, double y);
int oequal_int(int x, int y);
int ogreater_int(int x, int y);

#endif