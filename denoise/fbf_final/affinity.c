/*
 * Copyright (c) 2016, Anmol Popli <anmol.ap020@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * @file affinity.c
 * @brief  computes the number of physical cores on the system 
 * from the system cpu info and set threads accordingly
 *
 * @author ANMOL POPLI <anmol.ap020@gmail.com>
 **/

#ifdef __linux__
#include "headersreq.h"

int core_affinity();

/**
 * \brief Compute the number of physical cores and assign one thread to each
 * \return number of physical cores
 *
 * This routine computes the number of physical cores on
 * the system from the system cpu info.
 * It then sets number of threads equal to the number of
 * cores and inside a parallel region assigns the threads
 * to hyperthreads on distinct physical cores.
 */
#ifdef _OPENMP
    int core_affinity() {
        /** \brief Physical cores */
        int num_cores = 0;
        /** \brief Hyperthreads */
        int num_procs = omp_get_num_procs();
        bool proc_array[num_procs];
        bool core_array[num_procs];
        int i,j;
        for (i=0; i<num_procs; i++) {
            proc_array[i] = false;
            core_array[i] = false;
        }
        
        /* Read core-ids of hyperthreads from system info and mark one hyperthread from each physical core */
        FILE* myfile;
        for (i=0; i<num_procs; i++) {
            int core_id;
            char filename[50];
            sprintf(filename,"/sys/devices/system/cpu/cpu%d/topology/core_id",i);
            myfile = fopen(filename,"r");
    	    if (myfile == NULL) {
		return num_procs;
    	    }
            fscanf(myfile,"%d",&core_id);
            fclose(myfile);
            if (core_array[core_id]==false) {
                proc_array[i] = true;
                core_array[core_id] = true;
                num_cores++;
            }
        }
        
        /* Fork a parallel region and assign one thread to each physical core */
        int tid, procid;
        omp_set_num_threads(num_cores);
        i=0;
        #pragma omp parallel private(j) shared(num_procs,proc_array,i)
        {
            #pragma omp critical
            {
                j=1;
                while(j>0) {
                    if (proc_array[i]==true) {
                        cpu_set_t my_set;
                        CPU_ZERO(&my_set);
                        CPU_SET(i,&my_set);
                        sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
                        j=0;
                    }
                    i++;
                }
            }       
        }      
        return num_cores;
    }
#else
    int core_affinity() {
        return 0; /* Return number of cores as zero when OPENMP is not supported */
    }
#endif
#endif
