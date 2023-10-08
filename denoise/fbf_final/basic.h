/**
 * \file basic.h
 * \brief Portable types, math constants, and timing
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * This purpose of this file is to improve portability.
 *
 * Types \c uint8_t, \c uint16_t, \c uint32_t should be defined as
 * unsigned integer types such that
 * \li \c uint8_t  is 8-bit,  range 0 to 255
 * \li \c uint16_t is 16-bit, range 0 to 65535
 * \li \c uint32_t is 32-bit, range 0 to 4294967295
 *
 * Similarly, \c int8_t, \c int16_t, \c int32_t should be defined as
 * signed integer types such that
 * \li \c int8_t  is  8-bit, range        -128 to +127
 * \li \c int16_t is 16-bit, range      -32768 to +32767
 * \li \c int32_t is 32-bit, range -2147483648 to +2147483647
 *
 * These definitions are implemented with types \c __int8, \c __int16,
 * and \c __int32 under Windows and by including stdint.h under UNIX.
 *
 * To define the math constants, math.h is included, and any of the
 * following that were not defined by math.h are defined here according
 * to the values from Hart & Cheney.
 * \li M_2PI     = 2 pi       = 6.28318530717958647692528676655900576
 * \li M_PI      = pi         = 3.14159265358979323846264338327950288
 * \li M_PI_2    = pi/2       = 1.57079632679489661923132169163975144
 * \li M_PI_4    = pi/4       = 0.78539816339744830961566084581987572
 * \li M_PI_8    = pi/8       = 0.39269908169872415480783042290993786
 * \li M_SQRT2   = sqrt(2)    = 1.41421356237309504880168872420969808
 * \li M_1_SQRT2 = 1/sqrt(2)  = 0.70710678118654752440084436210484904
 * \li M_E       = e          = 2.71828182845904523536028747135266250
 * \li M_LOG2E   = log_2(e)   = 1.44269504088896340735992468100189213
 * \li M_LOG10E  = log_10(e)  = 0.43429448190325182765112891891660508
 * \li M_LN2     = log_e(2)   = 0.69314718055994530941723212145817657
 * \li M_LN10    = log_e(10)  = 2.30258509299404568401799145468436421
 * \li M_EULER   = Euler      = 0.57721566490153286060651209008240243
 * \li M_SQRT2PI = sqrt(2 pi) = 2.50662827463100050241576528481104525
 *
 * For precise timing, a function millisecond_timer() is defined.
 *
 *
 * Copyright (c) 2010-2013, Pascal Getreuer
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under, at your option, the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version, or the terms of the
 * simplified BSD license.
 *
 * You should have received a copy of these licenses along with this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef _BASIC_H_
#define _BASIC_H_

#include <math.h>
#include <stdlib.h>

/* Portable integer types */
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)

    /* Windows system: Use __intN types to define uint8_t, etc. */
    typedef unsigned __int8 uint8_t;
    typedef unsigned __int16 uint16_t;
    typedef unsigned __int32 uint32_t;
    typedef __int8 int8_t;
    typedef __int16 int16_t;
    typedef __int32 int32_t;
    
#else

    /* UNIX system: Use stdint to define uint8_t, etc. */
    #include <stdint.h>

#endif


/* Math constants (Hart & Cheney) */
#ifndef M_2PI
/** \brief The constant 2 pi */
#define M_2PI       6.28318530717958647692528676655900576
#endif
#ifndef M_PI
/** \brief The constant pi */
#define M_PI        3.14159265358979323846264338327950288
#endif
#ifndef M_PI_2
/** \brief The constant pi/2 */
#define M_PI_2      1.57079632679489661923132169163975144
#endif
#ifndef M_PI_4
/** \brief The constant pi/4 */
#define M_PI_4      0.78539816339744830961566084581987572
#endif
#ifndef M_PI_8
/** \brief The constant pi/8 */
#define M_PI_8      0.39269908169872415480783042290993786
#endif
#ifndef M_SQRT2
/** \brief The constant sqrt(2) */
#define M_SQRT2     1.41421356237309504880168872420969808
#endif
#ifndef M_1_SQRT2
/** \brief The constant 1/sqrt(2) */
#define M_1_SQRT2   0.70710678118654752440084436210484904
#endif
#ifndef M_E
/** \brief The natural number */
#define M_E         2.71828182845904523536028747135266250
#endif
#ifndef M_LOG2E
/** \brief Log base 2 of the natural number */
#define M_LOG2E     1.44269504088896340735992468100189213
#endif
#ifndef M_LOG10E
/** \brief Log base 10 of the natural number */
#define M_LOG10E    0.43429448190325182765112891891660508
#endif
#ifndef M_LN2
/** \brief Natural log of 2  */
#define M_LN2       0.69314718055994530941723212145817657
#endif
#ifndef M_LN10
/** \brief Natural log of 10 */
#define M_LN10      2.30258509299404568401799145468436421
#endif
#ifndef M_EULER
/** \brief Euler number */
#define M_EULER     0.57721566490153286060651209008240243
#endif

#ifndef M_SQRT2PI
/** \brief The constant sqrt(2 pi) */
#define M_SQRT2PI   2.50662827463100050241576528481104525
#endif

/** \brief Round double X */
#define ROUND(X) (floor((X) + 0.5))

/** \brief Round float X */
#define ROUNDF(X) (floor((X) + 0.5f))


#ifdef __GNUC__
    #ifndef ATTRIBUTE_UNUSED
    /** \brief Macro for the unused attribue GNU extension */
    #define ATTRIBUTE_UNUSED __attribute__((unused))
    #endif
    #ifndef ATTRIBUTE_ALWAYSINLINE
    /** \brief Macro for the always inline attribue GNU extension */
    #define ATTRIBUTE_ALWAYSINLINE __attribute__((always_inline))
    #endif
#else
    #define ATTRIBUTE_UNUSED
    #define ATTRIBUTE_ALWAYSINLINE
#endif

/**
 * \brief Millisecond-precision timer function
 * \return Clock value in units of milliseconds
 *
 * This routine implements a timer with millisecond precision.  In order to
 * obtain timing at high resolution, platform-specific functions are needed:
 *
 *    - On Windows systems, the GetSystemTime function is used.
 *    - On POSIX systems, the gettimeofday function is used.
 *
 * Otherwise as a fallback, time.h time is used, and in this case
 * millisecond_timer() has only second accuracy.  Preprocessor symbols are
 * checked in attempt to detect whether the platform is POSIX or Windows and
 * defines millisecond_timer() accordingly.  A particular implementation can
 * be forced by defining USE_GETSYSTEMTIME, USE_GETTIMEOFDAY, or USE_TIME.
 */
unsigned long millisecond_timer();

#endif /* _BASIC_H_ */
