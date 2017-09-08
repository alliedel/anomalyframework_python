/* changenumbers.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LENGTH_ONE_LINE (50000)

int outputSameLineNumTimes = 2;

int main(int argc, char **argv) {
    FILE *f, *fout;
    char outputFilename[500];
    char oneLine[MAX_LENGTH_ONE_LINE];
    if (argc != 2) {
        printf("Usage: %s fileToChangeNumbers\n", argv[0]);
        exit(0);
    }
    f = fopen(argv[1], "r");
    if (!f) {
        printf("Could not open input file: %s\n", argv[1]);
        exit(0);
    }
    sprintf(outputFilename, "%s.out", argv[1]);
    fout = fopen(outputFilename, "w");
    if (!fout) {
        printf("Could not open output file: '%s'\n", outputFilename);
        exit(0);
    }

    char *firstSpace_ptr;
    int frameNumber = 1;
    int numLinesWritten = 0;
    while (fgets(oneLine, MAX_LENGTH_ONE_LINE - 1, f) != NULL) {
        //      printf("string was '%s'\n", oneLine);
        firstSpace_ptr = strchr(oneLine, ' ');
        if (firstSpace_ptr == NULL)
            break;  // must be out of data ?
        for (int kkk = 0; kkk < outputSameLineNumTimes; kkk++) {
            fprintf(fout, "%d%s", frameNumber, firstSpace_ptr);
            numLinesWritten++;
        }
        frameNumber++;
    }
    fclose(f);
    fclose(fout);
    printf("Wrote %d lines of %d different frame numbers to '%s'\n",
           numLinesWritten, frameNumber - 1, outputFilename);
    return (0);
}
