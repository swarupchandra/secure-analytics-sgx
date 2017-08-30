#include <stdio.h>
#include <math.h>

#include "Enclave.h"
#include "Enclave_t.h"
#include "kmeans.h"


void printClusters(struc_cluster *cls, uint32_t K, uint32_t raw_num_features) {
    for(int i=0; i<K; i++) {
        printf("Cluster %d : ", i);
        printf("Centroid : ");
        for(int j=0; j<raw_num_features; j++) {
            printf("%f ", cls[i].centroid[j]);
        }
        printf("\n");
        printf("Class Proportion : ");
        for(int j=0; j<cls[i].num_classes; j++) {
            printf("%d ", cls[i].class_prop[j]);
        }
        printf("\n");
        if(cls[i].empty) {
            printf("Empty.\n");
        } else {
            printf("Not empty\n");
        }
    }
}


// Compute distance between two points with num_raw_features.
double euclidean_distance(double *p1, double *p2, uint32_t num_raw_features) {
    double dist = 0;
    for(int i=0; i<num_raw_features; i++) {
        dist += pow((p1[i] - p2[i]),2);
    }
    return dist;
}


// M-Step: Recompute cluster centroid
void recomputeCentroid(struc_cluster *cls, uint32_t K, double **data, uint32_t num_raw_features) {
    //for each cluster
    for(int i=0; i<K; i++) {
        for(int j=0; j<num_raw_features; j++) {
            cls[i].centroid[j] = 0;
            if(cls[i].data_len > 0) {
                cls[i].empty = false;
                cls[i].centroid[j] = cls[i].clus_data_sum[j] / cls[i].data_len;
            } else {
                cls[i].empty = true;
            }            
        }
    }    
}

void reinitializeCluster(struc_cluster *cls, uint32_t K) {
    // initialize class proportions and data 
    for(int i=0; i<K; i++) {   
        for(int j=0; j<cls[i].num_classes; j++) {
            cls[i].class_prop[j] = 0;
        }
        cls[i].empty = true;
        cls[i].data_len = 0;
    }
    // printf("Clusters reinitialized.\n");
}

