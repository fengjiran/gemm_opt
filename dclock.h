//
// Created by richard on 5/6/23.
//

#ifndef GEMM_OPT_DCLOCK_H
#define GEMM_OPT_DCLOCK_H

#include <sys/time.h>
#include <ctime>

static double gtod_ref_time_sec = 0.0;

/* Adapted from the bl2_clock() routine in the BLIS library */

double dclock() {
    double the_time, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, nullptr);

    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double) tv.tv_sec;

    norm_sec = (double) tv.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + tv.tv_usec * 1.0e-6;

    return the_time;
}

#endif //GEMM_OPT_DCLOCK_H
