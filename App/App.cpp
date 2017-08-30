
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
  #include <ctime>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
 #include <math.h>

# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }
    
    if (idx == ttl)
        printf("Error: Unexpected error occurred.\n");
}

/* Initialize the enclave:
 *   Step 1: try to retrieve the launch token saved by last transaction
 *   Step 2: call sgx_create_enclave to initialize an enclave instance
 *   Step 3: save the launch token if it is updated
 */
int initialize_enclave(void)
{
    char token_path[MAX_PATH] = {'\0'};
    sgx_launch_token_t token = {0};
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    int updated = 0;
    
    /* Step 1: try to retrieve the launch token saved by last transaction 
     *         if there is no token, then create a new one.
     */
    /* try to get the token saved in $HOME */
    const char *home_dir = getpwuid(getuid())->pw_dir;
    
    if (home_dir != NULL && 
        (strlen(home_dir)+strlen("/")+sizeof(TOKEN_FILENAME)+1) <= MAX_PATH) {
        /* compose the token path */
        strncpy(token_path, home_dir, strlen(home_dir));
        strncat(token_path, "/", strlen("/"));
        strncat(token_path, TOKEN_FILENAME, sizeof(TOKEN_FILENAME)+1);
    } else {
        /* if token path is too long or $HOME is NULL */
        strncpy(token_path, TOKEN_FILENAME, sizeof(TOKEN_FILENAME));
    }

    FILE *fp = fopen(token_path, "rb");
    if (fp == NULL && (fp = fopen(token_path, "wb")) == NULL) {
        printf("Warning: Failed to create/open the launch token file \"%s\".\n", token_path);
    }

    if (fp != NULL) {
        /* read the token from saved file */
        size_t read_num = fread(token, 1, sizeof(sgx_launch_token_t), fp);
        if (read_num != 0 && read_num != sizeof(sgx_launch_token_t)) {
            /* if token is invalid, clear the buffer */
            memset(&token, 0x0, sizeof(sgx_launch_token_t));
            printf("Warning: Invalid launch token read from \"%s\".\n", token_path);
        }
    }
    /* Step 2: call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        if (fp != NULL) fclose(fp);
        return -1;
    }

    /* Step 3: save the launch token if it is updated */
    if (updated == FALSE || fp == NULL) {
        /* if the token is not updated, or file handler is invalid, do not perform saving */
        if (fp != NULL) fclose(fp);
        return 0;
    }

    /* reopen the file with write capablity */
    fp = freopen(token_path, "wb", fp);
    if (fp == NULL) return 0;
    size_t write_num = fwrite(token, 1, sizeof(sgx_launch_token_t), fp);
    if (write_num != sizeof(sgx_launch_token_t))
        printf("Warning: Failed to save launch token to \"%s\".\n", token_path);
    fclose(fp);
    return 0;
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}



/**
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% MANAGER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
**/


//Base directory of input data files.
std::string filename; 

std::vector<uint8_t*> data_store;
std::vector<uint32_t> data_len;
double acc = 0;
double elapsed_sec = 0;


/**
Read data from file.
Data starting from start_data index, upto size datasize.
**/
void ocall_read_data(uint32_t start_data, uint32_t datasize, char *data, uint32_t len) {  
    std::ifstream f (filename);

    if(f.is_open()) {
        std::string str;
        std::string str_out = "";
        int count = 0;
        bool firstline = true;

        while(std::getline(f,str)) {
            if(firstline) {
                //ignore the header line.
                firstline = false;
            } else {
                //start reading from start_data
                if(count >= start_data) {
                    str_out  += str + ";";
                }
                ++count;
            }

            //Stop when datasize lines are read.
            if(count == (start_data + datasize)) {
                break;
            }
        }
        
        memcpy(data, str_out.c_str(), str_out.length() + 1);

        f.close();
    }
}


data_prop* getDataProperties() {
    std::ifstream f (filename);
    data_prop *prop = new data_prop;
    prop->num_data = 0;
    prop->num_features = 0;
    prop->num_classes = 0;

    if(f.is_open()) {
        std::string str;
        int count = 0;        

        std::getline(f,str);
        std::istringstream ss(str);
        std::string token;
        
        std::vector<int> features;
        while(std::getline(ss, token, ',')) {
            features.push_back(std::atoi(token.c_str()));
        }

        prop->num_data = features.at(0);
        prop->num_features = features.at(1);
        prop->num_classes = features.at(2);
        
        f.close();
    }

    return prop;
}

/**
Record classification accuracy from enclave.
**/
void ocall_print_acc(double ret_acc) {
    acc += ret_acc;
}

////////////////////////////////////////////////////////////////////////

//SAVE SEALED DATA

uint32_t ocall_write_file(uint8_t *buffer, uint32_t buflen, const char * filename) {
    FILE* pFile;
    pFile = fopen(filename, "a+");
    if (pFile) {
        fwrite(buffer, 1, buflen, pFile);
        fwrite("\n", 1, 1, pFile);
        printf("\nBuffer Written:%d\n", buflen);
        //for(int j=0; j< buflen ;j++){
        //    printf("%x ", (char*)buffer[j]);
        //}

    } else{
        printf("\nCan't open file");
        return 1;
    }
    printf("\n Encrypted data written successfully");
    fclose(pFile);
    return 0;
}



////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


/**
Simulating data analytics.
Training also performed within SGX as a surrogate. But, only record execution time of testing.
**/
void ocall_manager(uint32_t iter_size, uint32_t fake_size, uint32_t k_size) {
    data_prop *prop = getDataProperties();

    
    /** Types code
    Decision Tree
        11: Naive.
        12: Oblivious.
        13: Randomized.

    Naive Bayes
        21: Naive.
        22: Oblivious.
        23: Randomized.

    K-Means
        31: Naive.
        32: Oblivious.
        33: Randomized.
    **/

    //Example: Naive decision tree
    uint32_t type = 11;

    //First training a model.
    startModelTraining(global_eid, prop->num_data, prop->num_features, prop->num_classes, iter_size, fake_size, k_size, type);
    //Then start evaluation on trained model.
    //Begin simulation of evaluation on cloud environment.
    clock_t begin = std::clock();
    startStreamTesting(global_eid, prop->num_data, prop->num_features, prop->num_classes, iter_size, fake_size, k_size, type);
    clock_t end = std::clock();
    elapsed_sec  += double(end - begin) / CLOCKS_PER_SEC;

    printf("END OF APPLICATION.\n");
    delete prop;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////



/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    std::string basedir = "/data/"; // Data files directory

    std::string files[4] = {"syn003", "fc", "arrhythmia", "defaulter"}; // f: File names
    std::ofstream outfile; // output file

    uint32_t iter_size[4] = {16, 32, 64, 128}; // r: Data chunk size for streaming classification.
    uint32_t fake_size[3] = {2,4,8}; // p: For SGXRand (=L/N)
    uint32_t k_size[5] = {10,20,30,40,50}; // k: cluster size for k-means

    //EXAMPLE
    int i = 0; // selecting the first of all settings

    filename = basedir + files[i] +".csv";
    std::string outfilename = files[i] + "_sgx.out";
    
    // Time evaluation over 5 iterations.
    for(int j=0; j<5; j++) {

        /* Initialize the enclave */
        if(initialize_enclave() < 0){
            printf("Enter a character before exit ...\n");
            getchar();
            return -1; 
        }
    
        /* Utilize edger8r attributes */
        edger8r_array_attributes();
        edger8r_pointer_attributes();
        edger8r_type_attributes();
        edger8r_function_attributes();
        
        /* Utilize trusted libraries */
        ecall_libc_functions();
        ecall_libcxx_functions();
        ecall_thread_functions();

        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ocall_manager(iter_size[i], fake_size[i], k_size[i]);
        
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        /* Destroy the enclave */
        sgx_destroy_enclave(global_eid);
        
        printf("Info: SampleEnclave successfully returned.\n");
    }
    printf("Total Time=%f\n", elapsed_sec );
    printf("Total Accuracy=%f\n", acc);
    outfile.open(outfilename, std::ios_base::app);
    outfile << "k=" << k_size[i] << "\tf=" << fake_size[i] << "\ti=" << iter_size[i] << "\tTime=" << (elapsed_sec) << "\tAccuracy=" << (acc) << "\n";
    outfile.close();    
    acc = 0;
    elapsed_sec = 0;

    return 0;
}


