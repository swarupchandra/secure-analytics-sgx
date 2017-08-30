#include <stdio.h>
#include <cstring>

#include <sgx_trts.h>
#include <sgx_tseal.h>
#include "sgx_tcrypto.h"
#include "Enclave.h"
#include "Enclave_t.h"


/////////////////////////////////////////////////////////

// OBLIVIOUS PRIMITIVES

int omax_int(int x, int y ) {
    int decoy;
    if(x > y) {
        decoy = 1;
    } else {
        decoy = 0;
    }
    return (x*decoy + y*(1-decoy));
}

double omax_double(double x, double y) {
    double decoy;
    if(x > y) {
        decoy = 1;
    } else {
        decoy = 0;
    }
    return (x*decoy + y*(1-decoy));
}

int oequal_int(int x, int y) {
    int decoy;
    if(x == y) {
        decoy = 1;
    } else {
        decoy = 0;
    }
    return decoy;
}

int ogreater_int(int x, int y) {
    int decoy;
    if(x > y) {
        decoy = 1;
    } else {
        decoy = 0;
    }
    return decoy;
}


/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
//////// Batcher Odd-Even Merge Sort ////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////

void rand_exchange(int **data, int num_features, int x[], int a, int b) {
    int tmp = x[a];
    x[a] = x[b];
    x[b] = tmp;
    for(int i=0; i<num_features; i++) {
        tmp = data[a][i];
        data[a][i] = data[b][i];
        data[b][i] = tmp;
    }
}


void obliv_oddeven_sort(int **data, int num_features, int x[], int n) {
    for(int i=0; i<n; i++) {
        if(i & 1) { // 'i' is odd
            for(int j=2; j<n; j+=2) {     
                if (x[j] < x[j-1]) {
                    rand_exchange(data, num_features, x, j-1, j);
                }
            }
        } else {  
            for (int j=1; j<n; j+=2) {
                if (x[j] < x[j-1]) {
                    rand_exchange(data, num_features, x, j-1, j);
                }
            } 
        }
    }
}

void rand_exchange(double **data, int num_features, int x[], int a, int b) {
    int tmp = x[a];
    x[a] = x[b];
    x[b] = tmp;
    for(int i=0; i<num_features; i++) {
        double tmpd = data[a][i];
        data[a][i] = data[b][i];
        data[b][i] = tmpd;
    }
}


void obliv_oddeven_sort(double **data, int num_features, int x[], int n) {
    for(int i=0; i<n; i++) {
        if(i & 1) { // 'i' is odd
            for(int j=2; j<n; j+=2) {     
                if (x[j] < x[j-1]) {
                    rand_exchange(data, num_features, x, j-1, j);
                }
            }
        } else {  
            for (int j=1; j<n; j+=2) {
                if (x[j] < x[j-1]) {
                    rand_exchange(data, num_features, x, j-1, j);
                }
            } 
        }
    }
}

///////////////////////////////////////////////////////////

void obvi_exchange_data(int **data, int num_features, int x[], int a, int b) {
    int tmp = x[a];
    x[a] = x[b];
    x[b] = tmp;

    for(int i=0; i<num_features; i++) {
        tmp = data[a][i];
        data[a][i] = data[b][i];
        data[b][i] = tmp;
    }
}


void obliv_compare(int **data, int num_features, int x[], int a, int b) {
    int decoy_if = ogreater_int(x[a], x[b]);
    int ex = b*decoy_if + a*(1-decoy_if);
    obvi_exchange_data(data, num_features, x, a, ex);
}

void nonobliv_merge(int **data, int num_features,int x[], int lo, int n, int r) {
	int step = r * 2;
    if (step < n) {
		nonobliv_merge(data, num_features, x, lo, n, step);
		nonobliv_merge(data, num_features, x, lo+r, n, step);
		for (int i = lo+r; i+r < lo+n; i=i+step) {
			obliv_compare(data, num_features, x, i, i+r);
		}
	} else {
        obliv_compare(data, num_features, x, lo, lo+r);		
	}
}

void nonobliv_sort(int **data, int num_features, int x[], int lo, int n) {
    if (n > 1) {
		int mid = n / 2;
		nonobliv_sort(data, num_features, x, lo, mid);
		nonobliv_sort(data, num_features, x, lo+mid, mid);
		nonobliv_merge(data, num_features, x, lo, n, 1);
    }
}

//////////////////////////////////////////////////////////////////

// Random shuffle of a given array of length k within SGX Enclave;
// Non oblivious
void get_random_shuffle(int *list, int k) {
    int val;
    sgx_read_rand((unsigned char*) &val, 4);
    
    //Fisher-Yates shuffle algorithm
    for(int i=k-1; i>0; i--) {
        int r = (uint32_t)val%i;
        int temp = list[i];
        list[i] = list[r];
        list[r] = temp;
    }
}

// randomize data indices within the data matrix
void randomize_data_index(int **data, int num_data, int num_features) {
    //create a random list of shuffled data.
    int *random = new int[num_data];
    for(int i=0; i<num_data; i++) {
        random[i] = i;
    }
    get_random_shuffle(random, num_data);

    obliv_oddeven_sort(data, num_features, random, num_data);
    delete [] random;
}

int get_random_int(int k) {
    int val;
    sgx_read_rand((unsigned char*) &val, 4);
    if(k == 0) {
	return 0;
    } else {
	return (uint32_t)val%k;
    }
}

double get_random_double(double max, double min) {
    uint32_t val;
    sgx_read_rand((unsigned char*) &val, 4);
    if(max*10000 <1) {
        return 0.0;
    }
    int r = (uint32_t) val % (int)(max*10000);
    return (double) r/10000.0;
}

// randomize data indices within the data matrix
void randomize_rand_index(int **data, int num_data, int num_features) {
    //create a random list of shuffled data.
    int *random = new int[num_data];
    for(int i=0; i<num_data; i++) {
        random[i] = get_random_int(num_data);
    }
    obliv_oddeven_sort(data, num_features, random, num_data);
    delete [] random;
}

void randomize_rand_index(double **data, int num_data, int num_features) {
    //create a random list of shuffled data.
    int *random = new int[num_data];
    for(int i=0; i<num_data; i++) {
        random[i] = get_random_int(num_data);
    }
    obliv_oddeven_sort(data, num_features, random, num_data);
    delete [] random;
}



/////////////////////////////////////////////////////////////////////

//NOTE: These encryption and decryption routine is used within the Analytics (currently commented out).
// Uncomment the corresponding function calls to enable these cryptographic routines.

sgx_aes_gcm_128bit_tag_t mac;
uint8_t encrypted_buffer[12800];
const sgx_aes_gcm_128bit_key_t key[] = {'1','2','3','4','5','6','7','8','9','0','1','2','3','4','5','6'};
const uint8_t iv[] = {'1','2','3','4','5','6','7','8','9','0','1','2'};

// TODO: This code is called to encrypt results after classification.
void encrypt_data(uint8_t data[], size_t length){
    uint32_t err;
     sgx_status_t stat = sgx_rijndael128GCM_encrypt(key, data, length, encrypted_buffer, iv, 12, NULL, 0, &mac);
     printf("STATUS: %d\nENCRYPTED MESSAGE: %s\n",stat,(char*)encrypted_buffer);
     ocall_write_file(&err, encrypted_buffer, length, "out1.data"); // This is used to write encrypted result into a file
}

//This code is called to decrypt encrypted data copied to enclave.
void decrypt_data(uint8_t encrypted_buffer[], size_t length){
    uint8_t decrypt_buffer[length];
    sgx_status_t stat1 = sgx_rijndael128GCM_decrypt(key, encrypted_buffer, length, decrypt_buffer, iv, 12, NULL, 0, &mac);
    printf("\nSTATUS: %d \nDECRYPTED MESSAGE: %s\n",stat1,(char*)decrypt_buffer);
}



///////////////////////////////////////////////////////////////


void extract_features(int row_count, int **data, char *data_str, int datasize, int num_features) {
    char *end_str;
    char *data_point = strtok_r(data_str, ";", &end_str);

    while(data_point != NULL) {
        int col_count = 0;
        char *end_attr;
        char *data_point_attr = strtok_r(data_point, ",", &end_attr);
        while(data_point_attr != NULL) {
            int attr = atoi(data_point_attr);
            data[row_count][col_count] = attr;

            data_point_attr = strtok_r(NULL, ",", &end_attr);
            ++col_count;
            if(col_count >= num_features) {
                break;
            }
        }
        data_point = strtok_r(NULL, ";", &end_str);
    }
}

void extract_features(int row_count, double **data, char *data_str, int datasize, int num_features) {

    char *end_str;
    char *data_point = strtok_r(data_str, ";", &end_str);

    while(data_point != NULL) {
        int col_count = 0;
        char *end_attr;
        char *data_point_attr = strtok_r(data_point, ",", &end_attr);
        while(data_point_attr != NULL) {
            double attr = atof(data_point_attr);
            data[row_count][col_count] = attr;
            
            data_point_attr = strtok_r(NULL, ",", &end_attr);
            ++col_count;
            if(col_count >= num_features) {
                break;
            }
        }
        data_point = strtok_r(NULL, ";", &end_str);
    }
}

/////////////////////////////////////////////////////////////////////

//perform oblivious check 
int oblivious_check(bool *rows, int index) {
    int decoy_if;
    if(rows[index]) {
        decoy_if = 1;
    } else {
        decoy_if = 0;
    }
    return decoy_if;
}

