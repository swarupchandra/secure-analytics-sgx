
#include <stdarg.h>
#include <stdio.h>      /* vsnprintf */
#include <cstring>
#include <math.h>

#include <vector>
#include <queue>
#include <string>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */

#include "dt_obliv.h"
#include "dt_rand.h"
#include "dt_sgx.h"

#include "km_obliv.h"
#include "km_rand.h"
#include "km_sgx.h"

#include "nb_obliv.h"
#include "nb_rand.h"
#include "nb_sgx.h"

#include "analytics_util.h"


/* 
 * printf: 
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
}


void startModelTraining(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t fake_size, uint32_t K_size, uint32_t type) {
    printf("\n\nStarting training inside Enclave\n");

    switch(type) {
        case 11: 
            startDTTraining(num_data, num_features, num_classes, iter_size);
            break;
        case 12:
            startDTOblivTraining(num_data, num_features, num_classes, iter_size);
            break;
        case 13:
            startDTRandTraining(num_data, num_features, num_classes, iter_size, fake_size);
            break;
        case 21:
            startNBTraining(num_data, num_features, num_classes, iter_size);
            break;
        case 22:
            startNBOblivTraining(num_data, num_features, num_classes, iter_size);
            break;
        case 23:
            startNBRandTraining(num_data, num_features, num_classes, iter_size, fake_size);
            break;
        case 31:
            startKMTraining(num_data, num_features, num_classes, iter_size, K_size);
            break;
        case 32:
            startKMOblivTraining(num_data, num_features, num_classes, iter_size, K_size);
            break;
        case 33:
            startKMRandTraining(num_data, num_features, num_classes, iter_size, fake_size, K_size);
            break;

    }
}


void startStreamTesting(uint32_t num_data, uint32_t num_features, uint32_t num_classes, uint32_t iter_size, uint32_t fake_size, uint32_t K_size, uint32_t type) {
    printf("\n\nStarting testing inside Enclave\n");

    switch(type) {
        case 11: 
            startDTTesting(num_data, num_features, num_classes, iter_size);
            break;
        case 12:
            startDTOblivTesting(num_data, num_features, num_classes, iter_size);
            break;
        case 13:
            startDTRandTesting(num_data, num_features, num_classes, iter_size, fake_size);
            break;
        case 21:
            startNBTesting(num_data, num_features, num_classes, iter_size);
            break;
        case 22:
            startNBOblivTesting(num_data, num_features, num_classes, iter_size);
            break;
        case 23:
            startNBRandTesting(num_data, num_features, num_classes, iter_size, fake_size);
            break;
        case 31:
            startKMTesting(num_data, num_features, num_classes, iter_size, K_size);
            break;
        case 32:
            startKMOblivTesting(num_data, num_features, num_classes, iter_size, K_size);
            break;
        case 33:
            startKMRandTesting(num_data, num_features, num_classes, iter_size, fake_size, K_size);
            break;

    }    
    
}
