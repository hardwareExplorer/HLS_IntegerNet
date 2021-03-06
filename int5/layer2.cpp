#ifndef LAY2
#define LAY2


#include<ap_int.h>
#include "E:\MSbR\ResearchWork\Handson\Thesis_work\vivado_int5_testing\parameters.hpp"


static float gamma_2_int[16] = { 1.1728497 , 0.82948995, 1.049652  , 0.89543915, 0.96540076,
        1.1628076 , 0.94902384, 1.0643913 , 1.0018207 , 0.9267317 ,
        0.96516633, 1.0414768 , 0.9706925 , 1.009933  , 1.0074066 ,
        0.95611453};

static float beta_2_int[16] = {0.0318849 ,  0.01529017, -0.02720842,  0.16421649,  0.10866419,
         0.03586086, -0.1300936 , -0.14252795, -0.16946325, -0.14214069,
         0.21546532, -0.05591697,  0.06039352,  0.09177298,  0.05753997,
        -0.05247447 };

static float mean_2_int[16] = {78.00981 ,  92.774704,  49.1139  , 141.73747 , 122.78331 ,
         53.914177, 120.42949 ,  33.396145,  45.39722 , 160.5752  ,
        143.7633  , 113.35655 , 229.28181 ,  55.567074,  72.01832 ,
        131.27275 };

static float variance_2_int[16] = {28659.309 ,   8392.574 ,   5399.5146,  35683.402 ,  18951.338 ,
         20423.006 ,  41899.336 ,   5273.975 ,  12287.295 ,  20661.822 ,
         41419.844 ,  10714.73  , 102253.61  ,  12997.558 ,  10562.754 ,
         49534.22  };

static ap_int<INT_SIZE+2> wt2_int[32][16][1][3] =
{{{{ 25,  13,  20}},

        {{-18,  16, -19}},

        {{  0,  -5, -11}},

        {{  0,  26, -15}},

        {{-32,   3,  30}},

        {{ -1,  30,  20}},

        {{-16,  17, -24}},

        {{ 25,  -1,  10}},

        {{-18,  31, -19}},

        {{  9,  28, -15}},

        {{-18,   2, -20}},

        {{ 20,  -6,  16}},

        {{ 12,  27,   8}},

        {{ 31,  27, -12}},

        {{ -9,  -8, -13}},

        {{ 17, -30,   0}}},


       {{{ 17,  12, -20}},

        {{ 15, -22,   4}},

        {{ 29, -28,  -2}},

        {{-17,  26, -29}},

        {{ 17,  21,  20}},

        {{ -2,   2,  12}},

        {{ 20,   8, -10}},

        {{ 13,  29, -31}},

        {{ 17, -19,   0}},

        {{-18,  19,   1}},

        {{ -1,  25,  22}},

        {{ 13,  15,  19}},

        {{ 12,  -1, -30}},

        {{  4, -15,  18}},

        {{ -2,  -7, -22}},

        {{-13,  21,  26}}},


       {{{ -1,  12, -22}},

        {{  8, -18,   2}},

        {{ -3,   2,  15}},

        {{ 13,   3, -22}},

        {{ 13, -24, -27}},

        {{ 18, -29, -29}},

        {{  8,  15,  -6}},

        {{-30, -31, -32}},

        {{ 24,  17, -24}},

        {{ 19, -16,  18}},

        {{ 30,  -2,  31}},

        {{-29, -12, -14}},

        {{-19,   6, -21}},

        {{ 13,  -3,  17}},

        {{-12, -12,  30}},

        {{-28,  13,  16}}},


       {{{-16,  31,  23}},

        {{  5, -30,   5}},

        {{ -5,  13,  -6}},

        {{ 16, -15,   8}},

        {{ 10,  -7,   9}},

        {{ 21, -17, -24}},

        {{ -5, -29,   7}},

        {{  4,  -3,  27}},

        {{ 32,  30, -22}},

        {{-21,  -7, -22}},

        {{ 15, -11, -29}},

        {{ 10, -16,  24}},

        {{-18,  17,  24}},

        {{ 28, -11,  17}},

        {{ 17, -20,  21}},

        {{-15, -31, -17}}},


       {{{ -9, -27, -19}},

        {{ -1,   1,  -5}},

        {{ -3, -25,  29}},

        {{ 18, -22,  -2}},

        {{-25, -29, -11}},

        {{ 31,  18,   3}},

        {{ -6, -26,   6}},

        {{ 24, -11,  22}},

        {{-14,  30,  16}},

        {{ 27,  17,   2}},

        {{ 18,  17,   0}},

        {{ 12, -24,  11}},

        {{-19, -17,   3}},

        {{ 22,  -8, -16}},

        {{  6,  -4, -29}},

        {{ 24,  16,  10}}},


       {{{ 28,  26,  -3}},

        {{ 28,   4, -31}},

        {{-24, -31, -30}},

        {{-23,  -9, -16}},

        {{ 29,  22,  16}},

        {{-12,  27, -11}},

        {{-28,  -4, -28}},

        {{  4,   6, -27}},

        {{ 18, -12,   0}},

        {{ 28,   5, -11}},

        {{-14,   8,  -3}},

        {{-28,  -2,   6}},

        {{-18,  26,  30}},

        {{ 26, -20,   3}},

        {{-13,  28,  28}},

        {{-11,  22,  27}}},


       {{{ -5,  28,  -1}},

        {{  6, -32, -22}},

        {{-13, -31, -24}},

        {{ 22,  12, -12}},

        {{  3,   1,  11}},

        {{ 20, -12,  19}},

        {{ -7, -16, -18}},

        {{-20,   5, -11}},

        {{ 32, -26,  30}},

        {{-10,  16,  27}},

        {{-26,  24,  27}},

        {{  8, -31, -16}},

        {{ 20,  -3,   7}},

        {{-10,  -5,  16}},

        {{-28,  -7, -30}},

        {{ 25,  27,  24}}},


       {{{ -7,   6,   9}},

        {{ 31,  12, -18}},

        {{ -2, -23,   8}},

        {{-18,   7, -21}},

        {{-20, -30, -28}},

        {{-17,  21,  11}},

        {{ 22,  16,  -7}},

        {{ 13,  21, -14}},

        {{ 32, -12,  10}},

        {{-16,  21,  28}},

        {{ -1,  29,   0}},

        {{-26,  31,  16}},

        {{ 25, -29,  30}},

        {{ -6,  -5, -24}},

        {{  2,   8, -13}},

        {{ 18, -25,  -4}}},


       {{{ 18, -21,   3}},

        {{ 13,  -7,   3}},

        {{  4,  -3,  17}},

        {{ -2,  26,  12}},

        {{ -8,  21,  25}},

        {{ 32,  19,  -6}},

        {{ 18,  26,  10}},

        {{ -9,   0,  -8}},

        {{ 14,   7, -19}},

        {{  6,  11,   4}},

        {{-24,  -6, -21}},

        {{-13, -31, -24}},

        {{-17,  -7, -31}},

        {{-17,  31, -10}},

        {{ 21,   0, -27}},

        {{ 15, -16, -23}}},


       {{{-21, -14, -20}},

        {{ 30,  30,  -8}},

        {{ 31, -21,  24}},

        {{ 19,  -1,  14}},

        {{ 26, -17,  15}},

        {{  8,  30, -15}},

        {{ 13, -12,   6}},

        {{  3,  -5,  31}},

        {{  6,  -7, -16}},

        {{ -6, -24,   1}},

        {{  7, -24,  -8}},

        {{ -7,  20,  10}},

        {{  3, -19,   3}},

        {{-19,   7,  -7}},

        {{ 25,  17,  -8}},

        {{  7,  -6,   6}}},


       {{{-13, -24,   0}},

        {{ -9,   7, -31}},

        {{ -1,  25,  -4}},

        {{ 16,  -7, -26}},

        {{-10,  20,  23}},

        {{  3, -13,  -8}},

        {{-16,  20,   5}},

        {{ 27, -12, -22}},

        {{ 19,  25,  -1}},

        {{ 24,   3, -31}},

        {{  3,   2, -19}},

        {{  5,   9,  27}},

        {{  9, -21, -22}},

        {{-17,  31,  21}},

        {{-20,   6,  -1}},

        {{-20,  22,   4}}},


       {{{ 25,  -6, -31}},

        {{ 11,  20,  18}},

        {{ 12, -11,   4}},

        {{ 10,  -4,  20}},

        {{-14,   4, -21}},

        {{-15, -21,   4}},

        {{ -3,  21, -22}},

        {{ -4, -18,  -9}},

        {{-20,  29,  30}},

        {{ -2, -25,  15}},

        {{ 31,  27,  -3}},

        {{ -4,  29, -15}},

        {{  8, -28,   7}},

        {{-22, -26,  23}},

        {{ 23,  30,  12}},

        {{-26,   1,  30}}},


       {{{ -3,  32,   7}},

        {{-29,  24,  27}},

        {{-12,  11,  11}},

        {{ 20, -29,  12}},

        {{-28, -12, -22}},

        {{  5,  23, -26}},

        {{ 27, -10, -32}},

        {{-20,  -2, -17}},

        {{ 25, -26,  25}},

        {{  8,  26,  -3}},

        {{  0,   9,  27}},

        {{ 10, -26,  13}},

        {{-26,  13,  -4}},

        {{ 10, -20,  15}},

        {{ 22,   3,  -3}},

        {{ 27, -15, -19}}},


       {{{ 26,   3, -18}},

        {{ 18,  -7, -27}},

        {{ 14,   3,  13}},

        {{ 28, -17,   8}},

        {{-27,   2, -21}},

        {{ 27,  30,  18}},

        {{  1,  15,  15}},

        {{-20,  -1,  31}},

        {{-17,  17,  -9}},

        {{ 20,  14,  -3}},

        {{ 28, -20, -23}},

        {{-15,  19, -11}},

        {{-12,   3,  -8}},

        {{  1, -28,   9}},

        {{ -7, -32, -21}},

        {{  0,  17,  11}}},


       {{{ -7, -21, -16}},

        {{ 10,  -1, -21}},

        {{ -4, -32,  -6}},

        {{  9, -27, -31}},

        {{-19,  20,  16}},

        {{-13,  16, -31}},

        {{-12,  22,   7}},

        {{ -9,  -2, -22}},

        {{-13,  30,   0}},

        {{-22,  24,  25}},

        {{  4,   0, -12}},

        {{  0, -13, -28}},

        {{ 13, -14,   3}},

        {{ 16, -21, -17}},

        {{-27, -17, -24}},

        {{ 28, -10,  -2}}},


       {{{-17, -32,   4}},

        {{-24,  -8, -32}},

        {{-15,  -5,   0}},

        {{ 29, -11, -29}},

        {{ 32,  12,  -9}},

        {{ -3,   3,  16}},

        {{-20, -15, -18}},

        {{ 28,  -3, -14}},

        {{-14,  -4,  13}},

        {{-22,  -6, -20}},

        {{ 28, -12, -19}},

        {{-20, -29,   6}},

        {{ 17,  20,   6}},

        {{  2, -25,   4}},

        {{ 17, -26, -11}},

        {{-11,   6,  31}}},


       {{{-26, -27,  -5}},

        {{ -8, -19, -18}},

        {{-10,  -7,  14}},

        {{ 16,  13, -28}},

        {{ 22,  15, -26}},

        {{  9,  32,   2}},

        {{  8, -22,  -7}},

        {{ -6, -12,  22}},

        {{ 21,  19,  22}},

        {{  0,   8,   9}},

        {{  7,   2,  16}},

        {{-21,  12,  -8}},

        {{-10,   0,  22}},

        {{-29,  18, -12}},

        {{ -3,  20,   9}},

        {{ 26,  31,  25}}},


       {{{ 11,  10, -28}},

        {{-26,  18, -14}},

        {{-25,  -6, -11}},

        {{ -6,  23,  -4}},

        {{  7,  27, -12}},

        {{ 17,   1,  10}},

        {{ 18, -26, -13}},

        {{-24,  21,  10}},

        {{ 23, -27,  11}},

        {{-18, -25, -12}},

        {{-23,  19,  32}},

        {{ 25,  20, -21}},

        {{ 22, -16,  30}},

        {{  9, -30,  15}},

        {{-11,   5, -27}},

        {{-22,  22, -24}}},


       {{{ 32, -11, -22}},

        {{ 20,   5, -31}},

        {{ 24, -32,  10}},

        {{-13, -19,  -2}},

        {{ -4,   1,  -2}},

        {{-17,  14,   3}},

        {{  9,   9,   4}},

        {{ 31,  31, -12}},

        {{ 32,  20,   0}},

        {{  7,   0, -30}},

        {{-10,  -2,  22}},

        {{-14, -32,  -2}},

        {{ 21,  20, -32}},

        {{ 13,  23,  25}},

        {{  8, -22,  -6}},

        {{ 18,  -5,  24}}},


       {{{  4, -22,  12}},

        {{ -6, -24, -12}},

        {{ 20,   7, -23}},

        {{-26,  18, -28}},

        {{  7, -28, -20}},

        {{ 32,   0,  29}},

        {{-10,  -8,  14}},

        {{ 22,  32,  26}},

        {{  4,  22,  15}},

        {{ 23,  27,  21}},

        {{ -4,   3, -22}},

        {{ 22, -12,   4}},

        {{-24,  30,   6}},

        {{-14,  -7, -19}},

        {{-28,  -7,  -8}},

        {{ 24, -15,   9}}},


       {{{ 11,  24,  24}},

        {{  1, -27, -31}},

        {{  5, -27,   4}},

        {{-15,   1,   6}},

        {{-15,  21, -30}},

        {{ 31,  -3,  13}},

        {{ -4,  -8,  22}},

        {{ 28,  -7,  22}},

        {{  3, -13,  32}},

        {{ 21, -22, -31}},

        {{ -2,  16, -26}},

        {{ -2,  29,  -8}},

        {{ 19,  18, -11}},

        {{ 11,   3, -10}},

        {{ 12,  27,  10}},

        {{-21, -26,   8}}},


       {{{ 23,   1, -18}},

        {{  0,  -1,  16}},

        {{  6,  31,  -2}},

        {{-20,   9, -28}},

        {{ -7,   9,  15}},

        {{  1,  -9, -29}},

        {{-29, -21,  -3}},

        {{  0, -20,  30}},

        {{-29, -30,  10}},

        {{ 19,  31, -13}},

        {{ 23,  18, -23}},

        {{  8,  23,  32}},

        {{ 23, -21,  25}},

        {{ 28,  22,  20}},

        {{ 22, -31,   0}},

        {{ 29,   9,   0}}},


       {{{ -3, -14, -26}},

        {{-16, -27, -11}},

        {{ 26,   0, -27}},

        {{ 23, -18,  15}},

        {{ 18,  26,  24}},

        {{  7,  -2,  12}},

        {{ 24,  11, -20}},

        {{ -4, -10,  19}},

        {{-20,  -8, -21}},

        {{-26,  31, -29}},

        {{-15,  -8, -13}},

        {{-17,  10,  -7}},

        {{ 28,   3, -31}},

        {{-22,  27, -21}},

        {{ 14,  14,  13}},

        {{-13,  25,  17}}},


       {{{-12, -32,  21}},

        {{ -8,   5,  -2}},

        {{ -7, -19,   7}},

        {{ 14,   0,  28}},

        {{-12, -31, -30}},

        {{ 29, -17, -20}},

        {{ 19,  16,   2}},

        {{-14,  -4,  -3}},

        {{-27,  -1,  25}},

        {{ 31,  -5,   5}},

        {{-10,  -3,  14}},

        {{ 20,  24,  28}},

        {{ 22,   2,  -1}},

        {{-24, -29,  12}},

        {{ -3,  16,  26}},

        {{ -8, -31,  -8}}},


       {{{-10,  -5, -30}},

        {{  6,  30,   1}},

        {{-20,  12,  21}},

        {{  8,   8,   8}},

        {{ 13,  23,   6}},

        {{ -8,  -7, -22}},

        {{  4,   4,  13}},

        {{-18, -27,  26}},

        {{-20,   8,   6}},

        {{ 24, -23,  -6}},

        {{-23, -28,  10}},

        {{  8,  21,   0}},

        {{ 11,  -4, -17}},

        {{ 17,   4,  30}},

        {{-15,  17,  31}},

        {{ 20,   1,  -9}}},


       {{{ 19,   9,  19}},

        {{ -1,  10,  -4}},

        {{ -7,  -6,  20}},

        {{  2,   4,  25}},

        {{-19,  -3, -13}},

        {{-19, -12,  32}},

        {{-21, -24,  17}},

        {{ 31, -20,   9}},

        {{ -5,  -5,  22}},

        {{ -2,   3,  23}},

        {{-15,  21, -28}},

        {{ -2,  11, -22}},

        {{-14,  -7,  -9}},

        {{ 20,  15,  -4}},

        {{ 24,  -8,  28}},

        {{  7, -10, -31}}},


       {{{-22,  27,  10}},

        {{-32, -10,   5}},

        {{ 10,  27,  -4}},

        {{ 14,   6, -14}},

        {{ 21,   5, -27}},

        {{ 28,   7, -27}},

        {{ 14,  32, -24}},

        {{ 14, -18,  25}},

        {{-13, -29, -19}},

        {{ 24,   2,  15}},

        {{-12,   3,  20}},

        {{ 26,  26, -20}},

        {{-15,  29,  -3}},

        {{ 12,   8,  17}},

        {{  9,  18,  28}},

        {{ 10,  -6, -13}}},


       {{{-27, -18,  29}},

        {{-21, -10, -14}},

        {{  3,  16, -19}},

        {{ 15,   6,  12}},

        {{  7, -17, -23}},

        {{ 13,  27, -25}},

        {{-15,  -8,   5}},

        {{ 13, -25,  16}},

        {{-11, -32, -26}},

        {{ 31,  27, -18}},

        {{  3,   7, -16}},

        {{  3, -14,   8}},

        {{ 22,  27,  -4}},

        {{  9, -14, -12}},

        {{  9,  19,  -9}},

        {{ 15,  -8,  18}}},


       {{{-25,   1, -13}},

        {{  7,  -7,  14}},

        {{ 32,  10,  26}},

        {{  0,  22,   6}},

        {{ 15,  28,   1}},

        {{  1,   5, -24}},

        {{-25,  18, -14}},

        {{ 12,  26,  -3}},

        {{  7,  24,   3}},

        {{ 25, -20,  31}},

        {{-24, -27,  24}},

        {{ -8, -22,  11}},

        {{ 23, -25,  -3}},

        {{-16,  -9, -11}},

        {{ 12,  12,  26}},

        {{  3, -28, -23}}},


       {{{-24, -22,  -2}},

        {{ 11,  15,  15}},

        {{ 17,  27,  21}},

        {{ 27,   3, -27}},

        {{-32,  20,   7}},

        {{-26, -10,   3}},

        {{-20, -32, -26}},

        {{ 13,  -7, -32}},

        {{-32,  -4,  12}},

        {{-30, -24,  -8}},

        {{ 12,   3,  19}},

        {{ -6,  12, -18}},

        {{ 24, -21,   8}},

        {{ 32,  13,  21}},

        {{ -3, -15,  17}},

        {{-27,  -6, -14}}},


       {{{-17,   0,  14}},

        {{-14,   7, -10}},

        {{ 27,  16,  24}},

        {{-10,  11, -12}},

        {{  2,  23,   4}},

        {{-24,   0, -29}},

        {{  7, -18,  17}},

        {{ 24,  22,   7}},

        {{  2, -30,  10}},

        {{ 19,  20, -23}},

        {{ -7,  15,   6}},

        {{ 16, -13,  10}},

        {{-27, -31,  20}},

        {{-22,   7,   1}},

        {{-14,  23,  15}},

        {{  7,  -1,  19}}},


       {{{ 13,  26,   4}},

        {{  5,   0,   1}},

        {{-22,   1,  22}},

        {{-26,  26,  32}},

        {{-14, -16, -16}},

        {{  7, -21,  -7}},

        {{-11,  -5, -14}},

        {{-18,  24,  12}},

        {{ 17,  -2,   3}},

        {{  9, -14, -22}},

        {{-24, -13,   3}},

        {{  2,  22,  32}},

        {{ 26, -11,   5}},

        {{ -5,   3, -10}},

        {{  0,   7,   8}},

        {{-25,  -6,  11}}}}; 



static float scalar2[32] = {0.03099921, 0.03062909, 0.03093323, 0.03124818, 0.03032087,
       0.03041341, 0.0311418 , 0.03104547, 0.0311485 , 0.03044417,
       0.03037153, 0.03015895, 0.03112424, 0.03090826, 0.03087636,
       0.03123066, 0.03085412, 0.03125   , 0.03120723, 0.03108039,
       0.03118747, 0.03091955, 0.03038424, 0.03097915, 0.03002981,
       0.03113482, 0.03114993, 0.03079212, 0.03099015, 0.03118495,
0.03051103, 0.03124128};

#endif