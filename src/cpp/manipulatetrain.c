/* manipulatetrain.c */

/* reads in a .train file, and does stuff with it...this
   is just a test file -- to quicky manipulate / read
   .train files.
   Change whatever you'd like -- just put a comment up
   here, and use a good descriptor when committing (so 
   we can find old versions when needed in the repository)

   Reads 1 line from the file at a time -- doesn't attempt to read everything
   into one big array.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#include "../multicore-liblinear/linear.h"
#include <omp.h>

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
#define MAX_FEATURES (1000)
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))

/************************************************ */
/* our added vars go here: */
#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>

/* globals -- where we read in the file: */
double *frameNumber;
double *values;
int *indices;
int numColons;  // num features per line
int numLinesInFile;

double *values_oneColumn = NULL;

void ReadFile(char *fname);

void WriteOutput(const char *outFilename);

void WriteOutputOnlyStd(const char *outFilename);


static char *line = NULL;
static int max_line_len = 10000;

static char *readline(FILE *input) {
    int len;

    if (fgets(line, max_line_len, input) == NULL)
        return NULL;

    while (strrchr(line, '\n') == NULL) {
        max_line_len *= 2;
        line = (char *) realloc(line, max_line_len);
        len = (int) strlen(line);
        if (fgets(line + len, max_line_len - len, input) == NULL)
            break;
    }
    return line;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s trainfile\n", argv[0]);
        exit(0);
    }

    ReadFile(argv[1]);
    //  WriteOutput("manipulate.train");
    WriteOutputOnlyStd("manipulate.train");
} // end: main()


void ReadFile(char *fname) {
    FILE *fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "fopen(%s) FAILED\n", fname);
        perror("fopen");
        exit(1);
    } else {
        printf("opened '%s'\n", fname);
    }

    char *endptr;
    char *idx, *val, *label;
    line = Malloc(char, max_line_len);

    /* count number of lines: */
    numLinesInFile = 0;
    while (readline(fp))
        numLinesInFile++;
    numColons = 0;
    for (unsigned int iii = 0; iii < strlen(line); iii++) {
        if (line[iii] == ':')
            numColons++;
    }
    printf("numLinesInFile = %d\n", numLinesInFile);
    printf("numColons = %d\n", numColons);
    rewind(fp);

    /* allocate space to read all vars:  */
    frameNumber = Malloc(double, numLinesInFile);
    values = Malloc(double, numLinesInFile * numColons);
    indices = Malloc(int, numLinesInFile * numColons);
    int linenum = 0;
    int featureNum = 0;
    int index;

    while (readline(fp)) {
        label = strtok(line, " \t\n");
        if (label == NULL) {
            printf("line %d is an empty line.  exit()\n", linenum);
            exit(0);
        }
        frameNumber[linenum] = strtod(label, &endptr);
        if ((endptr == label) || (*endptr != '\0'))
            exit(0);

        featureNum = 0;
        while (1) {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if (val == NULL)
                break;

            index = linenum * numColons + featureNum;
            indices[index] = (int) strtol(idx, &endptr, 10);
            if (endptr == idx || *endptr != '\0')
                exit(0);

            values[index] = strtod(val, &endptr);
            if ((endptr == val) || (*endptr != '\0' && !isspace(*endptr)))
                exit(0);

            featureNum++;
        } // end: while(1) -- finished reading line
        if (featureNum != numColons) {
            printf("numFeatures for line %d was %d.  Which does not match numColons %d\n",
                   linenum, featureNum, numColons);
            exit(0);
        }
        linenum++;
    } // end: while(readline)
    fclose(fp);

    printf("Read %d lines,  %d features per line\n", linenum, numColons);

}


double standard_deviation(double *data, int n) {
    double mean = 0.0;
    double sum_deviation = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        mean += data[i];
    }
    mean = mean / n;
    for (i = 0; i < n; i++)
        sum_deviation += (data[i] - mean) * (data[i] - mean);
    return (sqrt(sum_deviation / n));
}

double GetNum(int line, int featurei) {
    double d;
    if (values_oneColumn == NULL)
        values_oneColumn = Malloc(double, numLinesInFile);


    int h = 10;
    int numFilled = 0;
    int i1 = MAX(0, line - h);
    int i2 = MIN(numLinesInFile - 1, line + h); // inclusive
    for (int linei = i1; linei <= i2; linei++) {
        values_oneColumn[linei - i1] = values[linei * numColons + featurei];
        numFilled++;
    }

    d = standard_deviation(values_oneColumn, numFilled);
    return (d);
}

void WriteOutput(const char *outFilename) {
    int index;
    double d;
    FILE *fout = fopen(outFilename, "w");
    if (!fout) {
        printf("Could not open output file: '%s'\n", outFilename);
        perror("fopen");
        exit(0);
    }

    for (int linenum = 0; linenum < numLinesInFile; linenum++) {
        fprintf(fout, "%.0f", frameNumber[linenum]);
        for (int fi = 0; fi < numColons; fi++) {
            index = linenum * numColons + fi;
            fprintf(fout, " %d:%g", indices[index], values[index]);
        }

        /* now we add an extra set of features, computed from those other ones: */
        for (int fi = 0; fi < numColons; fi++) {
            index = linenum * numColons + fi;
            d = GetNum(linenum, fi);
            fprintf(fout, " %d:%g", indices[index] + numColons, d);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);

    /* fout = fopen("stddev.txt", "w"); */
    /* /\* Compute std of feature 23 (matlab index 24) and print out: *\/ */
    /* for (int linenum = 0 ; linenum < numLinesInFile ; linenum++) */
    /*   { */
    /*     if (linenum == 48) */
    /*       printf("line 48\n"); */
    /*     d = GetNum(linenum, 23); */
    /*     fprintf(fout, "%g\n", d); */
    /*   } */
    /* fclose(fout); */
}


void WriteOutputOnlyStd(const char *outFilename) {
    int index;
    double d;
    FILE *fout = fopen(outFilename, "w");
    if (!fout) {
        printf("Could not open output file: '%s'\n", outFilename);
        perror("fopen");
        exit(0);
    }

    for (int linenum = 0; linenum < numLinesInFile; linenum++) {
        fprintf(fout, "%.0f", frameNumber[linenum]);
        /* replace features with stddev(features) */
        for (int fi = 0; fi < numColons; fi++) {
            index = linenum * numColons + fi;
            d = GetNum(linenum, fi);
            fprintf(fout, " %d:%g", indices[index], d);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
}

