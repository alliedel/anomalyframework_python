// score_shuffle.cpp
/* Description of primary variables used:
   struct problem prob:  Contains the 'problem'.
   prob.l is how many 'lines' in the .train file.
   prob.y is an double array of [prob.l] long
      When reading from the file, prob.y[] contains the frame number
      (which could be out of order in the file)
      When training, prob.y[] contains 0's or 1's.
   x_space[] is one big array where all the x data is stored.  That
      data can often be sparse, so this is not a "square" array -- 
      we allocate just enough space to contain all the data in
      the .train file, plus room for a -1 at the end, plus maybe
      some other bias term for each line in .train.
      x_space is a 'struct feature_node'.  Each of those contains
      which featureIndex it is, and the featureValue.
   prob.x[prob.l] is a set of pointers into x_space.  One pointer for
      each line in the .train file

   Other vars:
   originalFrameNumber -- int array of [prob.l] length.  As the data is
      read in, the 'frame numbers' (first element on each line) are saved
      in this array.  So originalFrameNumber[3] contains the frame number
      that started line line 4 of the .train file.

   summary_minFrameNumber - min frame number read in from .train
   summary_maxFrameNumber - max frame number read in from .train
   summary_numAllocated = summary_maxFrameNumber + 1
   summary_originalLineNumberForThisFrame : 
   An array from [0]..[summary_maxFrameNumber + 1] that tells you which
      line in the .train file contained this frame number.  Note that 
      if your train file contains image numbers 5000 to 9999, we will 
      allocate 10000 bytes even though the first 5000 will remain empty.
      Waste of space, but makes coding easier for now.  Might need to
      change if frame numbers aren't "near zero" in future .train files.
   summary_frameNumberExistedInInputFile[0..summary_maxFrameNumber + 1] -- 
      Tells how many time this frame existed in the .train file.
   summary_numTimesFilled[0..summary_maxFrameNumber + 1] -- will be updated
      by the threads as probababilities are computed/estimated.
   summary_probabilitySum[0..summary_maxFrameNumber + 1] -- Sum of all
      the probabilities for this frame -- updated by the threads.
*/

#include "multicore-liblinear/linear.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#include <vector>
#include <omp.h>
#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

/************************************************ */
/* our added vars go here: */
#include <dirent.h>   // opendir
#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <sys/stat.h>  // mkdir
#include <sys/types.h>
#include <unistd.h>
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MISC_STRING_LENGTH (500)
#define NUM_CLASSES (2)
//#define WINDOW_SIZE (10)
//#define WINDOW_STRIDE  (5)
#define PRINT_SECONDS (5)
#define INFO_FILE  ((char *) "info.m")
#define SUMMARY_FILE  ((char *) "summary.txt")
#define MEGAPLOT_FILE  ((char *) "megaplot.m")
#define FRAME_OUTPUT_FILE  ((char *) "frameorder.txt")
struct threadStartStopInfoStruct {  // only used in Luigi mode.
    int startFrame;
    int numSections;
    int numFramesPerSection;
    int firstValue; // how to fill y[0]...y[numFramesPerSection-1]
    int dropped;
};
std::vector<threadStartStopInfoStruct> threadStartStopInfo;
int numThreads = 1;  // can be overridden in the .runinfo file
int windowSize = 10; // can be overridden in the .runinfo file
int windowStride = 5;  // can be overridden in the .runinfo file
int maxBufferSize = -1;  // -1 means no maximum
char commandLine[1024];
int commandLine_argc;
char **commandLine_argv;
char output_directory_base[1024];  // where the output directories will be created
char output_directory_name[1024];  // output directory for each shuffle
char input_file_name_withpath[1024];  // moved here from main()
char input_file_name_base[1024];  // moved here from main()
//char model_file_name[1024];  // moved here from main()
//pthread_t *threads[NUM_THREADS];
pthread_t **threads;
int *frameNumber_original;  // we interpret 1st number in input file to be frame number...before we overwrite them.
int *frameNumber_shuffled;  // as we shuffle the .train lines, we shuffle frameNumber_original into this.
struct frameOrderInfoStruct {
    int startLineNumber;
    int endLineNumber; // INCLUSIVE!  (= startLineNumber + numOccurences - 1)
    int frameNumber;
    int numOccurences;
};
std::vector<frameOrderInfoStruct> frameOrderInfo;
sem_t summary_semaphore;
int summary_numAllocated;
int *summary_frameNumberExistedInInputFile; // really is HOW MANY TIMES IT EXISTS IN THE INPUT FILE
int *summary_originalLineNumberForThisFrame;
int *summary_numTimesFilled;
int *mega_summary_numTimesFilled;
double *summary_probabilitySum; // probability it's class '0'
double *mega_summary_probabilitySum; // probability it's class '0'
int summary_maxFrameNumber;
int summary_minFrameNumber;
int copyFirstLineOverAndOver = 0;
int justPrintWhatWouldRun = 0;
int featureIndicesAreZeroBased = 1;

/* New features being added 8/2017 */
int numShuffles = 0;  // 0 means just use input .train file
int blockShuffleSize = 1;  // if 'numShuffles' is non-zero
int reverseInput = 0;  // reverses data in the .train file.

int useAbsoluteValuesOfFeatures = 0;
int useSquareValuesOfFeatures = 0;
void *myThreadFunction(void * param);
void WriteInfoFile(void);
void WriteFrameOrderFile(void);
void UpdateSummary(int framenum, double predict_label, double *prob_estimates);
void WriteSummaryFile(void);
void WriteMegaSummaryFile(void);  // concatenation of all shuffled/reverse runs
void WriteMegaPlotFile(int numberOfRuns);  // Matlab script for quick plot of results
void ConcatenatePredictedFiles(void);
time_t GetSecondsInt(void);
double GetSeconds(void);
void CopyFirstLineOverAndOver(void);
void PrintProblem(struct problem *p);
//double GetSeconds(void);
static int ReadParameterFile(const char *fname);
void CreateOutputDirectory(int run_num);
/************************************************ */


void print_null(const char *s) {}

void exit_with_help()
{
    //  "Usage: train [options] training_set_file [model_file]\n"
    printf(
            "options:\n"
                    "-s type : set type of solver (default 1)\n"
                    "  for multi-class classification\n"
                    "	 0 -- L2-regularized logistic regression (primal)\n"
                    "	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
                    "	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
                    "	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
                    "	 4 -- support vector classification by Crammer and Singer\n"
                    "	 5 -- L1-regularized L2-loss support vector classification\n"
                    "	 6 -- L1-regularized logistic regression\n"
                    "	 7 -- L2-regularized logistic regression (dual)\n"
                    "  for regression\n"
                    "	11 -- L2-regularized L2-loss support vector regression (primal)\n"
                    "	12 -- L2-regularized L2-loss support vector regression (dual)\n"
                    "	13 -- L2-regularized L1-loss support vector regression (dual)\n"
                    "-c cost : set the parameter C (default 1)\n"
                    "-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
                    "-e epsilon : set tolerance of termination criterion\n"
                    "	-s 0 and 2\n"
                    "		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
                    "		where f is the primal function and pos/neg are # of\n"
                    "		positive/negative data (default 0.01)\n"
                    "	-s 11\n"
                    "		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
                    "	-s 1, 3, 4, and 7\n"
                    "		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
                    "	-s 5 and 6\n"
                    "		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
                    "		where f is the primal function (default 0.01)\n"
                    "	-s 12 and 13\n"
                    "		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
                    "		where f is the dual function (default 0.1)\n"
                    "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
                    "-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
                    "-v n: n-fold cross validation mode\n"
                    "-C : find parameter C (only for -s 0 and 2)\n"
                    "-n nr_thread : parallel version with [nr_thread] threads (default 1; only for -s 0, 2, 11)\n"
                    "-q : quiet mode (no outputs)\n"
    );
    exit(1);
}

void exit_input_error(int line_num)
{
    fprintf(stderr,"score_shuffle.cpp: Wrong input format at liness %d\n", line_num);
    exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
    int len;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

void parse_command_line(int argc, char **argv);
void read_problem(const char *filename);
void do_cross_validation();
void do_find_parameter_C();
void ReverseOriginalProblem(void);
void ShuffleProblem(struct problem *p_in, struct problem *p_out);
void CopyProblem(struct problem *p_in, struct problem *p_out);

struct feature_node *x_space;
size_t x_space_length; // how many bytes we allocated for x_space
struct parameter param;
struct problem prob_original;  // as read-in from the .train file.
struct problem prob;  // after shuffling/reversing. This is what the threads use
int flag_cross_validation;  // usually not set  (requires '-v')
int flag_find_C;
int flag_omp;
int flag_C_specified;
int flag_solver_specified;
int nr_fold;
double bias;
void MakeTestFileAndExit(void);

int main(int argc, char **argv)
{
    const char *error_msg;

    if (argc != 2) {
        printf("Usage: %s paramFile\n", argv[0]);
        printf("Command line is now inside 'paramFile'.  Options for the command line are:\n");
        exit_with_help();
        exit(0);
    }
    ReadParameterFile(argv[1]);
    parse_command_line(commandLine_argc, commandLine_argv);
    double read_tstart = GetSeconds();
    read_problem(input_file_name_withpath);  // fills global:  'prob_original'
    printf("%f seconds to read the file\n", GetSeconds() - read_tstart);
    /* rarely-used functions -- still supported, but won't be run for 
       backwards / shuffled sets -- only on the original input, then 
       they exit().  Probably not hard to change later....
    */
    if (flag_find_C)
      {
        do_find_parameter_C();
        exit(0);
      }
    else if(flag_cross_validation)
      {
        do_cross_validation();
        exit(0);
      }
    error_msg = check_parameter(&prob_original, &param);
    if(error_msg)
      {
        fprintf(stderr,"ERROR: %s\n",error_msg);
        exit(1);
      }
    if (param.nr_weight != 2) {
      printf("This program expects only -w0 and -w1.  Sorry\n");
      exit(0);
    }

    /* Unshuffled problem is in 'prob_original'.  The feature data is stored
       in x_space, pointed to by prob_original.x.  The x_space data will not be
       shuffled -- only the pointers
    */
    if (numThreads == 1)
      std::srand (1);
    else
      std::srand (unsigned(std::time(0)));  // seed random number generator
    threads = Malloc(pthread_t *, numThreads);
    prob.y = Malloc(double,prob_original.l);
    prob.x = Malloc(struct feature_node *,prob_original.l);
    frameNumber_shuffled = Malloc(int,prob_original.l);

    if (reverseInput)
      ReverseOriginalProblem();

    int runNumber = 0;
    int ret;
    while (1)
      {
        threadStartStopInfo.clear();
        sem_init(&summary_semaphore, 0, 1);

        if (numShuffles > 0)
          ShuffleProblem(&prob_original, &prob);
        else
          CopyProblem(&prob_original, &prob);
        CreateOutputDirectory(runNumber);  // also fills output_directory_name[]
        WriteInfoFile();
        WriteFrameOrderFile();
        for (int threadi = 0 ; threadi < numThreads ; threadi++)
          {
            threads[threadi] = new pthread_t;
            ret = pthread_create(threads[threadi], NULL, myThreadFunction,
                                 (void *) (intptr_t) threadi);
            if (ret != 0) {
              perror("pthread_create failed");
              exit(0);
            }
          }
        
        /* wait for each thread to finish */
        for (int threadi = 0 ; threadi < numThreads ; threadi++)
          pthread_join(*(threads[threadi]), NULL);

        WriteSummaryFile();
        ConcatenatePredictedFiles();
        printf("pthread_join: all threads finished for run %d\n", runNumber);
        /* clear out the summary structure: */
        for (int i = 0 ; i < summary_numAllocated ; i++) {
          summary_numTimesFilled[i] = 0;
          summary_probabilitySum[i] = 0.0;
        }
        runNumber++;
        if (numShuffles == 0)
          break;
        else if (runNumber >= numShuffles)
          break;
      } // end: while()
    WriteMegaSummaryFile();
    WriteMegaPlotFile(runNumber);

    sem_destroy(&summary_semaphore);
    destroy_param(&param);
    free(prob_original.y);
    free(prob_original.x);
    free(x_space);
    free(line);

    return 0;
} // end: main()


/* We usually don't use the '-C' parameter, so this routine usually isn't called.
   If the -C parameter is specified, the current code will run only on the
   original .train file -- NOT for each shuffle/reverse instance.
   So be warned
*/
void do_find_parameter_C()
{
    double start_C, best_C, best_rate;
    double max_C = 1024;
    if (flag_C_specified)
        start_C = param.C;
    else
        start_C = -1.0;
    printf("Doing parameter search with %d-fold cross validation.\n", nr_fold);
    find_parameter_C(&prob_original, &param, nr_fold, start_C, max_C, &best_C, &best_rate);
    printf("Best C = %lf  CV accuracy = %g%%\n", best_C, 100.0*best_rate);
}

/* We usually don't use the '-v' parameter, so this routine usually isn't called.
   If the -v parameter is specified, the current code will run only on the
   original .train file -- NOT for each shuffle/reverse instance.
   So be warned
*/
void do_cross_validation()
{
    int i;
    int total_correct = 0;
    double total_error = 0;
    double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
    double *target = Malloc(double, prob_original.l);

    cross_validation(&prob_original,&param,nr_fold,target);
    if(param.solver_type == L2R_L2LOSS_SVR ||
       param.solver_type == L2R_L1LOSS_SVR_DUAL ||
       param.solver_type == L2R_L2LOSS_SVR_DUAL)
    {
        for(i=0;i<prob_original.l;i++)
        {
            double y = prob_original.y[i];
            double v = target[i];
            total_error += (v-y)*(v-y);
            sumv += v;
            sumy += y;
            sumvv += v*v;
            sumyy += y*y;
            sumvy += v*y;
        }
        printf("Cross Validation Mean squared error = %g\n",total_error/prob_original.l);
        printf("Cross Validation Squared correlation coefficient = %g\n",
               ((prob_original.l*sumvy-sumv*sumy)*(prob_original.l*sumvy-sumv*sumy))/
               ((prob_original.l*sumvv-sumv*sumv)*(prob_original.l*sumyy-sumy*sumy))
        );
    }
    else
    {
        for(i = 0 ; i < prob_original.l ; i++)
            if(target[i] == prob_original.y[i])
                ++total_correct;
        printf("Cross Validation Accuracy = %g%%\n", 100.0 * total_correct / prob_original.l);
    }

    free(target);
}

void parse_command_line(int argc, char **argv)
{
    int i;
    void (*print_func)(const char*) = NULL;	// default printing to stdout

    // default values
    param.solver_type = L2R_L2LOSS_SVC_DUAL;
    param.C = 1;
    param.eps = INF; // see setting below
    param.p = 0.1;
    param.nr_thread = 1;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.init_sol = NULL;
    flag_cross_validation = 0;
    flag_C_specified = 0;
    flag_solver_specified = 0;
    flag_find_C = 0;
    flag_omp = 0;
    bias = -1;

    /* Allie change:  We presume 2 weights will be specified -- so allocate
       space for both.  The weight values will be filled in inside the thread code,
       and are dependent on the number of lines per training set
    */
    param.nr_weight = 2;
    param.weight_label = Malloc(int, param.nr_weight);
    param.weight = Malloc(double, param.nr_weight);
    param.weight_label[0] = 0;
    param.weight_label[1] = 1;
    param.weight[0] = 0.0; // will be overwritten before train()
    param.weight[1] = 0.0; // will be overwritten before train()

    // parse options
    for(i=0;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        if(++i>=argc)
            exit_with_help();
        switch(argv[i-1][1])
        {
            case 's':
                param.solver_type = atoi(argv[i]);
                flag_solver_specified = 1;
                break;

            case 'c':
                param.C = atof(argv[i]);
                flag_C_specified = 1;
                break;

            case 'p':
                param.p = atof(argv[i]);
                break;

            case 'e':
                param.eps = atof(argv[i]);
                break;

            case 'B':
                bias = atof(argv[i]);
                break;

            case 'n':
                flag_omp = 1;
                param.nr_thread = atoi(argv[i]);
                break;

            case 'w':
                printf("You should not use -w anymore.  2 weights are auto-allotted\n");
                exit(0);
                /* ++param.nr_weight; */
                /* param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight); */
                /* param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight); */
                /* param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]); */
                /* param.weight[param.nr_weight-1] = atof(argv[i]); */
                /* if (param.weight_label[param.nr_weight-1] != (param.nr_weight - 1)) { */
                /*   printf("your -w params are out of order  -w0 is first, -w1 is second, etc.\n"); */
                /*   exit(0); */
                /* } */
                /* break; */

            case 'v':
                flag_cross_validation = 1;
                nr_fold = atoi(argv[i]);
                if(nr_fold < 2)
                {
                    fprintf(stderr,"n-fold cross validation: n must >= 2\n");
                    exit_with_help();
                }
                break;

            case 'q':
                print_func = &print_null;
                i--;
                break;

            case 'C':
                flag_find_C = 1;
                i--;
                break;

            default:
                fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
                exit_with_help();
                break;
        }
    }

    set_print_string_function(print_func);

    if (i != argc) {
        printf("Your command line function seems to be mis-formed\n");
        exit_with_help();
        exit(0);
    }

    // default solver for parameter selection is L2R_L2LOSS_SVC
    if(flag_find_C)
    {
        if(!flag_cross_validation)
            nr_fold = 5;
        if(!flag_solver_specified)
        {
            fprintf(stderr, "Solver not specified. Using -s 2\n");
            param.solver_type = L2R_L2LOSS_SVC;
        }
        else if(param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC)
        {
            fprintf(stderr, "Warm-start parameter search only available for -s 0 and -s 2\n");
            exit_with_help();
        }
    }

    //default solver for parallel execution is L2R_L2LOSS_SVC
    if(flag_omp)
    {
        if(!flag_solver_specified)
        {
            fprintf(stderr, "Solver not specified. Using -s 2\n");
            param.solver_type = L2R_L2LOSS_SVC;
        }
        else if(param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC && param.solver_type != L2R_L2LOSS_SVR)
        {
            fprintf(stderr, "Parallel LIBLINEAR is only available for -s 0, 2, 11 now\n");
            exit_with_help();
        }
#ifdef CV_OMP
        omp_set_nested(1);
      omp_set_num_threads(nr_fold);
      printf("Total threads used: %d\n", nr_fold*param.nr_thread);
#else
        printf("Total threads used: %d\n", param.nr_thread);
#endif
    }

    if(param.eps == INF)
    {
        switch(param.solver_type)
        {
            case L2R_LR:
            case L2R_L2LOSS_SVC:
                param.eps = 0.01;
                break;
            case L2R_L2LOSS_SVR:
                param.eps = 0.001;
                break;
            case L2R_L2LOSS_SVC_DUAL:
            case L2R_L1LOSS_SVC_DUAL:
            case MCSVM_CS:
            case L2R_LR_DUAL:
                param.eps = 0.1;
                break;
            case L1R_L2LOSS_SVC:
            case L1R_LR:
                param.eps = 0.01;
                break;
            case L2R_L1LOSS_SVR_DUAL:
            case L2R_L2LOSS_SVR_DUAL:
                param.eps = 0.1;
                break;
        }
    }
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;

    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    prob_original.l = 0;
    elements = 0;
    max_line_len = 1024;
    line = Malloc(char,max_line_len);
    while(readline(fp)!=NULL)
    {
        char *p = strtok(line," \t"); // label

        // features
        while(1)
        {
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            elements++;
        }
        elements++; // for bias term
        prob_original.l++;
    }
    rewind(fp);
    /* note: 'elements' is the number of feature elements in the file -- it is NOT
       necessarily (prob_original.l * numElementsPerLine) -- e.g. the feature element indices
       could be 'sparse'....which train() apparently accepts just fine.
       The key is that the last .index variable for each line contain a -1:  that
       seems to be the indicator that a line has ended.
    */

    prob_original.bias = bias;
    prob_original.y = Malloc(double,prob_original.l);
    prob_original.x = Malloc(struct feature_node *,prob_original.l);
    x_space = Malloc(struct feature_node,elements+prob_original.l);
    int approxMemoryMalloced = (sizeof(double) * prob_original.l +
				sizeof(struct feature_node *) * prob_original.l +
				sizeof(struct feature_node) * (elements + prob_original.l));
    printf("approxMemoryMalloced = %d   (sizeof(feature_node) = %lu\n",
	   approxMemoryMalloced, sizeof(struct feature_node));
      
    x_space_length = elements + prob_original.l;
    max_index = 0;
    j=0;
    for(i=0;i<prob_original.l;i++)
    {
        inst_max_index = 0; // strtol gives 0 if wrong format
        readline(fp);
        prob_original.x[i] = &x_space[j];
        label = strtok(line," \t\n");
        if(label == NULL) { // empty line
            fprintf(stderr, "label is NULL; empty line?\n");
            exit_input_error(i + 1);
        }
        prob_original.y[i] = strtod(label,&endptr);
        if(endptr == label || *endptr != '\0') {
            fprintf(stderr, "endptr != label or *endptr != backslash-0\n");
            exit_input_error(i + 1);
        }

        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            if (featureIndicesAreZeroBased)
                x_space[j].index += 1;
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index) {
                fprintf(stderr, "Indexing error reading problem:\n");
                fprintf(stderr, "    endptr = %p, idx = %p\n", endptr, idx);
                fprintf(stderr, "    errno = %d\n", errno);
                fprintf(stderr, "    *endptr = %d\n", *endptr);
                fprintf(stderr, "    x_space[j].index = %d  vs.  inst_max_index = %d\n",
                        x_space[j].index, inst_max_index);
                exit_input_error(i+1);  // will not return
            }

            inst_max_index = x_space[j].index;
            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if (useAbsoluteValuesOfFeatures)
                x_space[j].value = fabs(x_space[j].value);
            if (useSquareValuesOfFeatures)
                x_space[j].value = fabs(x_space[j].value);

            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))){
                fprintf(stderr, "Input error 4\n");
                exit_input_error(i+1);
            }

            ++j;
        } // end: while (reading of a single line)

        if(inst_max_index > max_index)
            max_index = inst_max_index;

        if(prob_original.bias >= 0)
            x_space[j++].value = prob_original.bias;

        x_space[j++].index = -1;  /* By setting this to -1, train() will know that
                                   this is the end of this line of features.
                                   VERY important if the features provided are
                                   sparse (i.e. not all indices are specified in
                                   the input file)
                                */
    } // end: for(i < prob_original.l)

    if(prob_original.bias >= 0)
    {
        prob_original.n=max_index+1;
        for(i=1;i<prob_original.l;i++)
            (prob_original.x[i]-2)->index = prob_original.n;
        x_space[j-2].index = prob_original.n;
    }
    else
        prob_original.n=max_index;

    fclose(fp);

    if (copyFirstLineOverAndOver)
        CopyFirstLineOverAndOver();

    /* We are using the first element of the file as the frame number of the video.
       We overwrite these when we start running, so we save them all here first.
       Note: the 'y' values are actually read from the file as double's.  But we
       presume they are really 'int's....i.e. only round numbers of frames!
    */
    /* 8/2017: By writing this file out immediately after reading in the 
       problem from the train file, that means it will not properly reflect
       any 'shuffling' or 'reversing' that we've been asked to do while
       we run this data.   If you use FRAME_OUTPUT_FILE, you need to be aware
       of that -- it ONLY reflects the order of the frames in the .train file
       that was passed in!
    */
    frameNumber_original = Malloc(int,prob_original.l);
    summary_minFrameNumber = (int) prob_original.y[0];
    summary_maxFrameNumber = summary_maxFrameNumber;
    for (i = 0 ; i < prob_original.l ; i++) {
        frameNumber_original[i] = (int) prob_original.y[i];
        summary_minFrameNumber = MIN(summary_minFrameNumber, frameNumber_original[i]);
        summary_maxFrameNumber = MAX(summary_maxFrameNumber, frameNumber_original[i]);
        if (frameNumber_original[i] < 0) {
            printf("You have a negative frame number.  That makes you a bad person.\n");
            printf("Line %d.  Frame number is %f    exiting\n", i + 1, prob_original.y[i]);
            exit(0);
        }
    }
    printf("%d frames. minFrameNumber = %d; maxFrameNumber = %d\n", prob_original.l,
           summary_minFrameNumber, summary_maxFrameNumber);
    summary_numAllocated = summary_maxFrameNumber + 1;
    printf("summary_numAllocated = %d\n", summary_numAllocated);
    summary_frameNumberExistedInInputFile = Malloc(int, summary_numAllocated); // 0..max
    summary_numTimesFilled = Malloc(int, summary_numAllocated);   // 0..max
    mega_summary_numTimesFilled = Malloc(int, summary_numAllocated);   // 0..max
    summary_probabilitySum = Malloc(double, summary_numAllocated);   // 0..max
    mega_summary_probabilitySum = Malloc(double, summary_numAllocated);   // 0..max
    summary_originalLineNumberForThisFrame = Malloc(int, summary_numAllocated);
    for (i = 0 ; i < summary_numAllocated ; i++) {
        summary_frameNumberExistedInInputFile[i] = 0;
        summary_numTimesFilled[i] = 0;
        mega_summary_numTimesFilled[i] = 0;
        summary_probabilitySum[i] = 0.0;
        mega_summary_probabilitySum[i] = 0.0;
        summary_originalLineNumberForThisFrame[i] = -1;
    }

    int lastFrameNumber = -1;
    int thisFrameNumber;
    frameOrderInfo.clear();
    struct frameOrderInfoStruct my_frameOrderInfo;
    my_frameOrderInfo.frameNumber = -1;
    for (i = 0 ; i < prob_original.l ; i++) {
        thisFrameNumber = frameNumber_original[i];
        /* frame numbers can be duplicated, but they must be duplicated CONSECUTIVELY
           in the input .train file.  Check for that here, because if they are
           duplicated and scattered, it messes things up */
        if ((summary_frameNumberExistedInInputFile[thisFrameNumber] > 0) &&
            (thisFrameNumber != lastFrameNumber))
        {
            printf("**ERROR Your input file has NON-Consecutive Duplicate Frame Numbers\n");
            printf("        frameNumber %d on line %d was the first example\n", thisFrameNumber, i);
            printf("        It showed up %d times before that\n",
                   summary_frameNumberExistedInInputFile[thisFrameNumber]);
            printf("        Fix the input file.  exiting\n");
            exit(0);
        }
        summary_frameNumberExistedInInputFile[thisFrameNumber]++;
        // summary_originalLineNumberForThisFrame[frameNumber_original[i]] = i + 1;  // 1-based, per Allie -- before 8/2017
        summary_originalLineNumberForThisFrame[frameNumber_original[i]] = i;
        lastFrameNumber = thisFrameNumber;

        if (thisFrameNumber == my_frameOrderInfo.frameNumber)
        {
            // just increment the last entry:
            if (frameOrderInfo.size() == 0) {
                printf("Mark has a frameOrderInfo bug. Sorry. exit()\n");
                exit(0);
            }
            size_t iii = frameOrderInfo.size() - 1;
            frameOrderInfo[iii].numOccurences++;
            frameOrderInfo[iii].endLineNumber++;
        }
        else
        {
            /* need to add a new one: */
            my_frameOrderInfo.startLineNumber = i;  // 0.. (prob_original.l-1)
            my_frameOrderInfo.endLineNumber = i;
            my_frameOrderInfo.frameNumber = thisFrameNumber;
            my_frameOrderInfo.numOccurences = 1;
            frameOrderInfo.push_back(my_frameOrderInfo);
        }
    } // end: for(i)

    if (frameOrderInfo.size() == 0)
      {
	printf("frameOrderInfo.size() == 0!! That's bad.  prob_original.l = %d\n", prob_original.l);
	exit(0);
      }

    if (frameOrderInfo[frameOrderInfo.size() - 1].endLineNumber != (prob_original.l - 1)) {
        printf("frameOrderInfo buf: %d vs %d\n",
               frameOrderInfo[frameOrderInfo.size() - 1].endLineNumber, prob_original.l - 1);
        exit(0);
    }

    /* sanity check to make sure I counted right: */
    int sum = 0;
    for (unsigned int iii = 0 ; iii < frameOrderInfo.size() ; iii++)
        sum += frameOrderInfo[iii].numOccurences;
    if (sum != prob_original.l) {
        printf("Bug in creating frameOrderInfo. sum = %d, prob_original.l = %d\n", sum, prob_original.l);
        exit(0);
    }

    int printFrameOrderInfo = 0;
    if (printFrameOrderInfo) {
        printf("i\tline\tframe\tnumTimes\n");
        for (unsigned int iii = 0 ; iii < frameOrderInfo.size() ; iii++)
            printf("%d\t%d\t%d\t%d\n", iii, frameOrderInfo[iii].startLineNumber,
                   frameOrderInfo[iii].frameNumber, frameOrderInfo[iii].numOccurences);
    }

    /* go through all of them to see if any were duplicated */
    for (i = 0 ; i < summary_numAllocated ; i++) {
        if (summary_frameNumberExistedInInputFile[i] > 1)
            printf("DUPLICATE Frame number: %d was in the file %d times\n",
                   i, summary_frameNumberExistedInInputFile[i]);
    }

#if 0
    /* printout stuff for confirming if things work: */
  printf("Debug printing for frame numbers\n");
  printf("For each line read:\n");
  for (i = 0 ; i < prob_original.l ; i++) {
    printf("  Line %d was frame %d\n", i, frameNumber_original[i]);
  }

  printf("For all lines indices allotted:\n");
  for (i = 0 ; i < summary_numAllocated ; i++) {
    if (summary_frameNumberExistedInInputFile[i])
      printf("  frame num %d EXISTED in original file\n", i);
    else
      printf("  frame num %d did not exist in original file\n", i);
  }
  exit(0);
#endif
} // end: read_problem()


/* copies from 'prob' in the outerblock -- which was already read in from
   the file by their stuff */
void CopyProb(struct problem *p)
{
    p->l = prob.l;
    p->n = prob.n;
    p->bias = prob.bias;
    p->y = Malloc(double, prob.l);  // allocate my own space for y
    for (int i = 0 ; i < prob.l ; i++)
    {
        p->y[i] = prob.y[i];
    }
#if 1
    p->x = prob.x;  /* ALLIE says this is "ok"...because they don't change.
		     If they do change, we're pretty screwed.  But this will
		     save memory */
#else
    /* if we had to copy them over, it would look something like this: */
  /* UNTESTED CODE! */
  struct feature_node *my_x_space;
  p->x = Malloc(struct feature_node *, prob.l);
  my_x_space = Malloc(struct feature_node, x_space_length);
  for (size_t i = 0 ; i < x_space_length ; i++)
    my_x_space[i] = x_space[i];
  for (int i = 0 ; i < prob.l ; i++) {
    p->x[i] = &(my_x_space[i * (prob.n + 1)]);
    printf("x[%d] => my_x_space[%d]\n", i, i * (prob.n + 1));
  }
#endif
}

void CopyParam(struct parameter *p)
{
    p->solver_type = param.solver_type;
    p->eps = param.eps;
    p->C = param.C;
    p->nr_thread = param.nr_thread;
    p->nr_weight = param.nr_weight;
    p->p = param.p;

    p->weight_label = Malloc(int, param.nr_weight);
    p->weight = Malloc(double, param.nr_weight);
    for (int i = 0 ; i < param.nr_weight ; i++) {
        p->weight_label[i] = param.weight_label[i];
        p->weight[i] = param.weight[i];
    }
    p->init_sol = NULL;
}


/* All threads are allowed to call this routine.
   Since it *changes* a common set of variables, you must use a semaphore
   to ensure mutual exclusion to those vars.
*/
void UpdateSummary(int framenum, double predict_label, double *prob_estimates)
{
    sem_wait(&summary_semaphore);
    if ((framenum < summary_minFrameNumber) || (framenum > summary_maxFrameNumber)) {
        printf("UpdateSummary() called with framenum = %d.\n", framenum);
        printf("    Should have been in the range [%d..%d]\n", summary_minFrameNumber,
               summary_maxFrameNumber);
        printf("    exit()\n");
        exit(0);
    }
    // todo: Does 'predict_label' matter?
    summary_probabilitySum[framenum] += prob_estimates[1];
    mega_summary_probabilitySum[framenum] += prob_estimates[1];
    summary_numTimesFilled[framenum]++;
    mega_summary_numTimesFilled[framenum]++;
    sem_post(&summary_semaphore);
}



/* This is being changed to deal with duplicate, consective frame numbers.
   The definition of windowSize changes when you allow duplicates:
   windowSize = 3   means we will load the train() function with data
   from 3 different frames out of the .train file -- this may use many
   more than 10 of the lines.  e.g. If the train file has lines that start:
   1 2 2 3 3 3 4 4 4 4 5 5 5 5 5 6 6 6 6 6 6
   if windowSize = 3, the first data set will use all the lines with the
   first 3 frame numbers.  In this case, [1 2 2 3 3 3] -- that's 6 total lines.
*/
void *myThreadFunction(void * passedInFromPthreadCreate)
{
    FILE *output;
    intptr_t tmp_myThreadNumber = (intptr_t) passedInFromPthreadCreate;
    int myThreadNumber = (int) tmp_myThreadNumber;
    char model_output_fname[MISC_STRING_LENGTH];
    char prediction_output_fname[MISC_STRING_LENGTH];
    struct problem my_prob;  // all the data
    struct problem my_prob_for_train;
    struct parameter my_param;
    struct model* my_model;
    double prob_estimates[NUM_CLASSES];
    double predict_label;
    time_t lastPrintSeconds = GetSecondsInt();
    time_t deltaSeconds;
    int numFilesCreated = 0;

    CopyProb(&my_prob);
    CopyParam(&my_param);
    if (NUM_CLASSES != 2) {
        printf("You better have fixed the setting of the 0's and 1's below if this isn't 2.\n");
        exit(0);
    }

    int numUniqueFrameNumbers = (int) frameOrderInfo.size();
    int startNumber = windowSize * 2 + myThreadNumber * windowStride;
    printf("thread %d starting at %09d\n", myThreadNumber, startNumber);

    for (int i = 0 ; i < prob.l ; i++)  // must set ALL y's to 0...so we use prob.l NOT my_prob.l
        my_prob.y[i] = 0;

    int numFilesToGenerate = numUniqueFrameNumbers / (numThreads * windowStride);
    int startLine_of_zeros = 0;
    int startLine, endLine;
    int numLines;
    /*  "ufi" = Unique Frame Index.  It's an index into frameOrderInfo[]
        It's values can be from 0 .. (frameOrderInfo.size()-1)
    */
    for (int ufi = startNumber ; ufi <= numUniqueFrameNumbers ; ufi += (numThreads * windowStride))
    {
        /* this run will go from frameOrderInfo[0]...frameOrderInfo[ufi] */
        /* The LAST 'windowSize' unique frames get 1's.  Everything else is 0.  */

        /* ************** Set up vars for training ********************/
        for (int i = 0 ; i < prob.l ; i++)
            my_prob.y[i] = 0;  // overkill, but easy to make sure everything is 0 to start.

        startLine = frameOrderInfo[ufi - windowSize].startLineNumber;  // start of the 1's.
        endLine = frameOrderInfo[ufi - 1].endLineNumber;  // INCLUSIVE  -- end of the 1's.
	if (maxBufferSize == -1)
	  {
	    startLine_of_zeros = 0;  // there is no max...use all the X's & Y's
	    numLines = endLine + 1;
	  }
	else
	  {
	    startLine_of_zeros = MAX(0, endLine - maxBufferSize + 1);
	    numLines = MIN(endLine + 1, maxBufferSize);
	  }
        for (int i = startLine ; i <= endLine ; i++)
            my_prob.y[i] = 1;
        my_prob.l = numLines;
        //int numOneLines = (endLine - startLine + 1);
        //int numZeroLines = numLines - numOneLines;
#if 0
	/* the weight of the 1's is:  numberOfZeros / totalNumberOfLines */
        my_param.weight[1] = (double) (ufi - windowSize) / (double) ufi;
        my_param.weight[0] = 1.0 - my_param.weight[1];
#else
	/* Allie says in 8/2017: the weight[0] should be the number of ones divided
	   by the total number of (ones + zeros) passed in to train() */
        my_param.weight[0] = (double) (endLine - startLine + 1) / numLines;
	my_param.weight[1] = 1.0 - my_param.weight[0];
#endif
        my_param.weight[1] = my_param.weight[1]; // Place to scale if necessary
        my_param.weight[0] = my_param.weight[0]; // Place to scale if necessary
        if (justPrintWhatWouldRun) {
            printf("frameIndices: 0's: %d..%d  1's: %d..%d\n", startLine_of_zeros, ufi-windowSize-1, ufi-windowSize, ufi-1);
            printf("    line numbers: 0's: %d..%d  1's: %d..%d  (numLines %d)\n", startLine_of_zeros, startLine - 1, startLine, endLine, numLines);
	    printf("    startLine_of_zeros = %d\n", startLine_of_zeros);
	    printf("    weights: %f / %f\n", my_param.weight[0], my_param.weight[1]);
            //      printf("1's are lines %5d:%5d(INCLUSIVE) ufi %2d:%2d  weight[1]=%f    thread %d\n", startLine, endLine,
            //             ufi-windowSize, ufi-1, my_param.weight[1], myThreadNumber);
        }

        /* ************** Run train() *******************************************/
        // if (numThreads == 1)
        //            srand(1);   // if only 1 task, add this -- makes outputs consistent.
        sprintf(model_output_fname, "%s/%s_%09d.model", output_directory_name, input_file_name_base, numLines);
	my_prob_for_train.l = numLines;
	my_prob_for_train.n = prob.n;
	my_prob_for_train.bias = prob.bias;
	my_prob_for_train.y = my_prob.y + startLine_of_zeros;
	my_prob_for_train.x = my_prob.x + startLine_of_zeros;
	
        if (justPrintWhatWouldRun)
        {
            printf("    model_output_fname: %s\n", model_output_fname);
        }
        else
	  {
            // my_model = train(&my_prob, &my_param);  // todo:  change train() to accept a pointer to the _model
            my_model = train(&my_prob_for_train, &my_param);  // todo:  change train() to accept a pointer to the _model
            if (save_model(model_output_fname, my_model))
            {
                fprintf(stderr,"Thread %d failed writing model to '%s'.  exiting\n", myThreadNumber, model_output_fname);
                perror("save_model");
                exit(1);
            }
        }

        /* ************** Generate Predictions & write to file ********************/
        sprintf(prediction_output_fname, "%s/%s_%09d.predict", output_directory_name, input_file_name_base, numLines);
        if (justPrintWhatWouldRun)
        {
            printf("    prediction_output_fname: %s\n", prediction_output_fname);
        }
        else
        {
            output = fopen(prediction_output_fname, "wb");
            if (!output) {
                fprintf(stderr,"Thread %d failed to open '%s' for writing\n", myThreadNumber, prediction_output_fname);
                perror("fopen");
                exit(1);
            }
            int framenum;
            for (int iii = startLine ; iii <= endLine ; iii++)
            {
                predict_label = predict_probability(my_model, my_prob.x[iii],
                                                    prob_estimates);
                framenum = frameNumber_shuffled[iii];
                fprintf(output, "%d %g %g %g\n", framenum,
                        predict_label, prob_estimates[0], prob_estimates[1]);
                UpdateSummary(framenum, predict_label, prob_estimates);
            }
            fclose(output);
            free_and_destroy_model(&my_model);
        }

        numFilesCreated++;
        deltaSeconds = GetSecondsInt() - lastPrintSeconds;
        if (!justPrintWhatWouldRun && (deltaSeconds >= PRINT_SECONDS)) {
            lastPrintSeconds = GetSecondsInt();
            printf("thread %d: generated %d files out of %d\n", myThreadNumber, numFilesCreated,
                   numFilesToGenerate);
        }
    } // end: for(numLines)
    printf("thread %d ended after generating %d files\n", myThreadNumber, numFilesCreated);
    return(NULL);
} // end: myThreadFunction()


void MakeTestFileAndExit(void)
{
    FILE *f;
    f = fopen("test", "wb");
    if (!f) {
        perror("fopen");
        exit(0);
    }

    double num = 0;
    for (int i = 0 ; i < 100 ; i++) {
        fprintf(f, "0 ");
        for (int j = 1 ; j <= 40 ; j++) {
            fprintf(f, "%d:%.1f ", j, num);
            num += 1;
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("wrote 'test'\n");
    printf("exit\n");
    exit(0);
}


/* returns only the integer-portion of the current time in seconds */
time_t GetSecondsInt(void)
{
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return(t.tv_sec);
}

double GetSeconds(void)
{
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return(t.tv_sec + t.tv_nsec / 1e9);
}

/* Called before the threads are kicked-off....doesn't seem like it
   matters whether is uses the original .train file, or if it uses
   shuffled data.  I'd think it was better to be sure to use the
   NON-original data....just in case we add vars that are specific
   to each run (like if it's forward or backward)
*/
void WriteInfoFile(void)
{
    FILE *f;
    char fname[MISC_STRING_LENGTH];
    sprintf(fname, "%s/%s", output_directory_name, INFO_FILE);
    f = fopen(fname, "w");
    if (!f) {
        perror("WriteInfoFile could not open file");
        exit(0);
    }

    fprintf(f, "global allie_numThreadsRun;\n");
    fprintf(f, "global allie_windowSize;\n");
    fprintf(f, "global allie_windowStride;\n");
    fprintf(f, "global allie_inputFile;\n");
    fprintf(f, "global allie_inputFile_base;\n");
    fprintf(f, "global allie_numLinesInInputFile;\n");
    fprintf(f, "global allie_numVarsPerLine;\n");
    fprintf(f, "global allie_featureIndicesAreZeroBased;\n");
    fprintf(f, "global allie_blockShuffleSize;\n");
    fprintf(f, "global allie_reverseInput;\n");
    fprintf(f, "global allie_numShuffles;\n");
    fprintf(f, "global allie_maxBufferSize;\n");
    fprintf(f, "\n");
    fprintf(f, "allie_numThreadsRun = %d;\n", numThreads);
    fprintf(f, "allie_windowSize = %d;\n", windowSize);
    fprintf(f, "allie_windowStride = %d;\n", windowStride);
    fprintf(f, "allie_inputFile = '%s';\n", input_file_name_withpath);
    fprintf(f, "allie_inputFile_base = '%s';\n", input_file_name_base);
    fprintf(f, "allie_numLinesInInputFile = %d;\n", prob.l);
    fprintf(f, "allie_numVarsPerLine = %d;\n", prob.n - 1);
    fprintf(f, "allie_featureIndicesAreZeroBased = %d;\n", featureIndicesAreZeroBased);
    fprintf(f, "allie_blockShuffleSize = %d;\n", blockShuffleSize);
    fprintf(f, "allie_reverseInput = %d;\n", reverseInput);
    fprintf(f, "allie_numShuffles = %d;\n", numShuffles);
    fprintf(f, "allie_maxBufferSize = %d;\n", maxBufferSize);
    fprintf(f, "if (exist('allie_plot_results'))\n");
    fprintf(f, "  Z = load('summary.txt');\n");
    fprintf(f, "  framenum = Z(:,1);\n");
    fprintf(f, "  orig_linenum = Z(:,2);\n");
    fprintf(f, "  numPredicted = Z(:,3);\n");
    fprintf(f, "  sumPredicted = Z(:,4);\n");
    fprintf(f, "  x = Z(:,5);\n");
    fprintf(f, "  xx = x ./ (1-x);\n");
    fprintf(f, "  if (0)\n");
    fprintf(f, "    figure;\n");
    fprintf(f, "    plot(framenum, x, '.-');\n");
    fprintf(f, "    title 'x'\n");
    fprintf(f, "    xlabel 'frame number'\n");
    fprintf(f, "  end\n");
    fprintf(f, "  figure;\n");
    fprintf(f, "  plot(framenum, xx, '.-');\n");
    fprintf(f, "  title 'x / (1-x)'\n");
    fprintf(f, "  xlabel 'frame number'\n");
    fprintf(f, "end\n");

    fclose(f);
    printf("Wrote '%s'\n", fname);
}


/* One goes in each subdirectory -- this files contains the frame 
   numbers, in the order after shuffling/revsersing is complete.
*/
void WriteFrameOrderFile(void)
{
  char frame_output_fname[MISC_STRING_LENGTH];
  sprintf(frame_output_fname, "%s/%s", output_directory_name, FRAME_OUTPUT_FILE);
  FILE *fframeoutput = fopen(frame_output_fname, "w");
  if (!fframeoutput) {
    printf("Could not open '%s' for writing.  exit()\n", frame_output_fname);
    perror("frame output file");
    exit(0);
  }
  for (int i = 0 ; i < prob_original.l ; i++) 
    {
      fprintf(fframeoutput, "%d\n", frameNumber_shuffled[i]);
    }
  fclose(fframeoutput);
  printf("Wrote '%s'\n", frame_output_fname);
}


void WriteSummaryFile(void)
{
    FILE *f;
    char fname[MISC_STRING_LENGTH];
    sprintf(fname, "%s/%s", output_directory_name, SUMMARY_FILE);
    f = fopen(fname, "w");
    if (!f) {
        perror("WriteInfoFile could not open file");
        exit(0);
    }

    int num0 = 0;
    int num1 = 0;
    int num2 = 0;
    int num_other = 0;
    double avg;
    for (int i = 0 ; i < summary_numAllocated ; i++) {
        if (summary_frameNumberExistedInInputFile[i]) {
            if (summary_numTimesFilled[i])
                avg = summary_probabilitySum[i] / summary_numTimesFilled[i];
            else
                avg = 0.0;
            fprintf(f, "%d\t%d\t%d\t%9.6f\t%9.6f\n",
                    i, summary_originalLineNumberForThisFrame[i],
                    summary_numTimesFilled[i], summary_probabilitySum[i],
                    avg);
            if (summary_numTimesFilled[i] == 0)
                num0++;
            else if (summary_numTimesFilled[i] == 1)
                num1++;
            else if (summary_numTimesFilled[i] == 2)
                num2++;
            else
                num_other++;
        }
    }
    fclose(f);
    printf("Wrote '%s'\n", fname);
    printf("   %d frames never predicted\n", num0);
    printf("   %d frames predicted once\n", num1);
    printf("   %d frames predicted twice\n", num2);
    printf("   %d frames predicted more than twice\n", num_other);
} // end: WriteSummaryFile()

void WriteMegaSummaryFile(void)
{
  FILE *f;
  char fname[MISC_STRING_LENGTH];
  sprintf(fname, "%s/%s", output_directory_base, SUMMARY_FILE);
  f = fopen(fname, "w");
  if (!f) {
    perror("WriteMegaSummaryFile could not open file");
    exit(0);
  }

  int num0 = 0;
  int num1 = 0;
  int num2 = 0;
  int num_other = 0;
  double avg;
  for (int i = 0 ; i < summary_numAllocated ; i++) {
    if (summary_frameNumberExistedInInputFile[i]) {
      if (mega_summary_numTimesFilled[i])
        avg = mega_summary_probabilitySum[i] / mega_summary_numTimesFilled[i];
      else
        avg = 0.0;
      fprintf(f, "%d\t%d\t%d\t%9.6f\t%9.6f\n",
              i, summary_originalLineNumberForThisFrame[i],
              mega_summary_numTimesFilled[i], mega_summary_probabilitySum[i],
              avg);
      if (mega_summary_numTimesFilled[i] == 0)
        num0++;
      else if (mega_summary_numTimesFilled[i] == 1)
        num1++;
      else if (mega_summary_numTimesFilled[i] == 2)
        num2++;
      else
        num_other++;
    }
  }
  fclose(f);
  printf("Wrote MEGA summary file: '%s'\n", fname);
  printf("   %d frames never predicted\n", num0);
  printf("   %d frames predicted once\n", num1);
  printf("   %d frames predicted twice\n", num2);
  printf("   %d frames predicted more than twice\n", num_other);
} // end: WriteMegaSummaryFile()

void WriteMegaPlotFile(int numberOfRuns)
{
  FILE *f;
  char fname[MISC_STRING_LENGTH];
  sprintf(fname, "%s/%s", output_directory_base, MEGAPLOT_FILE);
  f = fopen(fname, "w");
  if (!f) {
    perror("WriteMegaSummaryFile could not open file");
    exit(0);
  }

  fprintf(f, "for i = 0:%d\n", numberOfRuns - 1);
  fprintf(f, "    sss = sprintf('%%06d/summary.txt', i);\n");
  fprintf(f, "    Z = load(sss);\n");
  fprintf(f, "    framenum = Z(:,1);\n");
  fprintf(f, "    x = Z(:,5);\n");
  fprintf(f, "    xx = x ./ (1-x);\n");
  fprintf(f, "    figure\n");
  fprintf(f, "    plot(framenum, xx, '.-');\n");
  fprintf(f, "    ttext = sprintf('%%06d:  x / (1-x)', i);\n");
  fprintf(f, "    title(ttext);\n");
  fprintf(f, "    xlabel ('frame number');\n");
  fprintf(f, "end\n");
  fprintf(f, "Z = load('summary.txt');\n");
  fprintf(f, "framenum = Z(:,1);\n");
  fprintf(f, "x = Z(:,5);\n");
  fprintf(f, "xx = x ./ (1-x);\n");
  fprintf(f, "figure\n");
  fprintf(f, "plot(framenum, xx, '.-');\n");
  fprintf(f, "title('MEGA:  x / (1-x)');\n");
  fprintf(f, "xlabel ('frame number');\n");
  fclose(f);
  printf("Wrote MEGA plot file: '%s'\n", fname);
} // end: WriteMegaPlotFile()


void ConcatenatePredictedFiles(void)
{
    char sss[MISC_STRING_LENGTH * 5];

    sprintf(sss, "cat %s/*.predict >> %s/everything.predict", output_directory_name, output_directory_name);
    printf("%s\n", sss);
    if (system(sss) == -1)
        printf("cat of .predict files FAILED\n");
}


static int GetLine(FILE *f, char *var, char *rightHandSide)
{
    char line[1000];
    if (fgets(line, 999, f) == NULL)
        return(-1);

    if (line[strlen(line)-1] == '\n')
        line[strlen(line)-1] = '\0'; // to make printing easier.

    if (!isalpha(line[0]))
    {
        var[0] = '#';
        var[1] = '\0';
        return(0);
    }

    //  int num = sscanf(line, "%100[^=]=%s", var, rightHandSide);
    int num = sscanf(line, "%100[^=]=%[^\n]", var, rightHandSide);
    if (num == 1)
    {
        printf("\n\n\n**** PARAMETER FILE PARSE ERROR. Aborting ****\n");
        printf("bad line: '%s'\n\n\n", line);
        printf("exit(0)\n");
        exit(0); // EXIT
    }
    else if (num == 0)
    {
        printf("might be parse-software-bug. don't think I should get here\n");
        printf("line: '%s'\n", line);
        var[0] = '#';
        var[1] = '\0';
        return(0);
    }
    char *pspace = strchr(var, ' ');
    if (pspace)
        *pspace = '\0';

    return(0);
} // end: GetLine() -- string version

static char *StripLeadingAndTrailingSpaces(char *str)
{
    char *end;
    while(isspace(*str)) str++;
    if(*str == 0)  // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace(*end)) end--;

    // Write new null terminator
    *(end+1) = 0;

    return str;
}

static int paramFileVerbose = 0;
#define DPRINT() if (paramFileVerbose) { printf("param:  %s = '%s'\n", var, rhs); }
//#define DPRINT() ;
static int ReadParameterFile(const char *filename)
{
    FILE *f = fopen(filename, "r");
    char var[MISC_STRING_LENGTH];
    char *rhs = (char *) malloc(MISC_STRING_LENGTH); // I don't free() this. sue me.
    int linenum = 0;
    int parseErrors = 0;
    int numgood = 0;

    /* set defaults here: */
    strcpy(output_directory_base, "./output/"); // default

    f = fopen(filename, "r");
    if (!f)
      {
        printf("ReadParameterFile('%s') FAILED\n", filename);
        perror("ReadParameterFile");
        exit(0);
      }
    while (GetLine(f, var, rhs) != -1)
    {
        rhs = StripLeadingAndTrailingSpaces(rhs);
        linenum++;
        if (var[0] == '#') {
            // ignore this line
        } else if (!strcmp(var, "commandLine")) {
            strcpy(commandLine, rhs);  DPRINT();
        } else if (!strcmp(var, "inputFile")) {
            strcpy(input_file_name_withpath, rhs);  numgood++;  DPRINT();
        } else if (!strcmp(var, "outputDirectory")) {
            strcpy(output_directory_base, rhs);  numgood++;  DPRINT();
        } else if (!strcmp(var, "numThreads")) {
            numThreads = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "windowSize")) {
            windowSize = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "windowStride")) {
	  windowStride = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "maxBufferSize")) {
	  maxBufferSize = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "copyFirstLineOverAndOver")) {
            copyFirstLineOverAndOver = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "useAbsoluteValuesOfFeatures")) {
            useAbsoluteValuesOfFeatures = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "useSquareValuesOfFeatures")) {
            useSquareValuesOfFeatures = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "justPrintWhatWouldRun")) {
            justPrintWhatWouldRun = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "featureIndicesAreZeroBased")) {
          featureIndicesAreZeroBased = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "blockShuffleSize")) {
          blockShuffleSize = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "reverseInput")) {
          reverseInput = atoi(rhs); numgood++;  DPRINT();
        } else if (!strcmp(var, "numShuffles")) {
          numShuffles = atoi(rhs); numgood++;  DPRINT();
        } else {
            if (parseErrors == 0)
                printf("\nPARAMETER FILE PARSE ERROR. File: '%s'\n", filename);
            printf("Line %d unknown var: '%s'\n", linenum, var);
            parseErrors++;
        }
    }; // end: while()
    fclose(f);
    if (parseErrors)
    {
        printf("Parameter file '%s' read.  %d vars set.  %d parse errors.\n",
               filename, numgood, parseErrors);
        printf("Fix the file and run again\n");
        printf("exit(0)\n");
        exit(0);
    }
    char *lastSlashPtr = strrchr(input_file_name_withpath, '/');
    if (!lastSlashPtr)
        strcpy(input_file_name_base, input_file_name_withpath);
    else
        strcpy(input_file_name_base, lastSlashPtr + 1);
    printf("inputFile = '%s'\n", input_file_name_withpath);
    printf("inputFile_base = '%s'\n", input_file_name_base);
    printf("commandLine = %s\n", commandLine);
    printf("outputDirectory = '%s'\n", output_directory_base);

    /* create the output directory, if it doesn't already exist: */
    DIR *dir = opendir(output_directory_base);
    if (dir)
      {
        closedir(dir);
        printf("Output directory exists.  Removing everything in it:\n");
        char sss[MISC_STRING_LENGTH * 5];
        sprintf(sss, "    rm -rf %s/*", output_directory_base);
        printf("%s\n", sss);
        if (system(sss) == -1) {
          printf("system(%s) FAILED\n", sss);
          exit(0);
        }
      }
    else 
      {
        if (mkdir(output_directory_base, S_IRWXU | S_IRWXG | S_IRWXO) != 0) 
          {
            printf("ERROR: Attempted to mkdir(%s)\n", output_directory_base);
            perror("  Can't create output directory.  exit()");
            exit(0);
          }
        printf("mkdir(%s) -- created successfully\n", output_directory_base);
      }

    /* copy the parameter file used for this run to the output directory: */
    char sss[MISC_STRING_LENGTH];
    sprintf(sss, "cp %s %s", filename, output_directory_base);
    printf("%s\n", sss);
    if (system(sss) == -1) {
        printf("system(%s) FAILED\n", sss);
        exit(0);
    }

    char *token;
    char *str = strdup(commandLine);
    commandLine_argc = 0;
    commandLine_argv = (char **) malloc(1000 * sizeof(char *)); // I don't free() this.
    while ((token = strtok(str, " ")))
    {
        str = NULL;
        commandLine_argv[commandLine_argc] = (char *) malloc(strlen(token) + 1);
        strcpy(commandLine_argv[commandLine_argc], token);
        commandLine_argc++;
    }
    return(0);
} // end: ReadParameterFile()


/* The output of each run is put in its own subdirectory:  000000, 000001, 000002, etc..
   The code expects output_directory_name to be set to that subdirectory
*/
void CreateOutputDirectory(int run_num)
{
  sprintf(output_directory_name, "%s/%06d", output_directory_base, run_num);
  if (mkdir(output_directory_name, S_IRWXU | S_IRWXG | S_IRWXO) != 0) {
    printf("ERROR: Attempted to mkdir(%s) for run number %d\n",
           output_directory_name, run_num);
    perror("  Can't create output directory.  exit()");
    exit(0);
  } else {
    printf("mkdir(%s) -- created successfully for run number %d\n",
           output_directory_name, run_num);
  }

}


/* Happens inside 'read_problem()' -- so needs to use the _original version of 'prob'
 */
void CopyFirstLineOverAndOver(void)
{
    /* leave prob_original.y alone (that's frame number).
       Overwrite all records with the contents of the first one.
       xspace[] is one big long array with
    */
    if ((x_space_length % prob_original.l) != 0) {
        printf("x_space_length = %ld;  prob_original.l = %d; Not an integer multiple. Bug?\n",
               x_space_length, prob_original.l);
        exit(0);
    }
    size_t numElementsInOneRow = x_space_length / prob_original.l;
    for (int i = 1 ; i < prob_original.l ; i++) {
        for (size_t j = 0 ; j < numElementsInOneRow ; j++) {
            x_space[i * numElementsInOneRow + j].index = x_space[j].index;
            x_space[i * numElementsInOneRow + j].value = x_space[j].value;
        }
    }
}


/* This function is not called -- when somebody puts it back into the 
   normally-running-code, you'll need to decide if it should use _original
   data, or regular 'prob'.
*/
void PrintProblem(struct problem *p)
{
    size_t numElementsInOneRow = x_space_length / prob.l;

    printf("PrintProblem:\n");
    printf("  %d lines;  n = %d;  numElementsInOneRow = %lu\n", p->l, p->n,
           numElementsInOneRow);
    printf("  bias = %f\n", p->bias);

    for (int i = 0 ; i < p->l ; i++)
    {
        printf("x[%d]:\n", i);
        for (size_t j = 0 ; j < numElementsInOneRow ; j++)
            printf("%d:%f  ", p->x[i][j].index, p->x[i][j].value);
        printf("\n");
    }

    for (int i = 0 ; i < p->l ; i++)
        printf("y[%d] = %f\n", i, p->y[i]);
} // end: PrintProblem()


/* Reverses the 'prob_original' variable -- but not quite like you'd think.
   Given the windowStride / windowSize, it figures out which points
   (at the end of the .train file) that would not have been used / computed.
   Those points stay where they are, and still won't be computed.
   The other points are reversed.
   e.g. We have 204 points.  windowSize is 5.  The last 4 points will never
   be computed.  So the reversed points are:  199, 198, 197, ... 1, 0, 200, 201, 202, 203
   We are not allocating new memory here -- we are actually reversing the numbers
   in prob_original.  So all the .y values get revsersed, and all the .x pointers
   get reversed.  Many ways of doing that, but mine is to copy, then reverse.
   (slow, but easily debuggable)
   We also reverse the frameNumber_original array.
*/
void ReverseOriginalProblem(void)
{
  /* uses the globals 'windowSize', and 'windowStride' and prob_original.l to determine
     what the last computed point will be.  Points after that will never be computed. */
  int numPointsNotUsedAtEnd = (prob_original.l - (windowSize * 2)) % windowStride;
  int lastPointUsedIndex = prob_original.l - numPointsNotUsedAtEnd - 1;
  if (lastPointUsedIndex <= 0)
    {
      printf("Tried reversing the problem, but something's wrong\n");
      printf("   Is the windowSize too big for this problem?\n");
      printf("   windowSize = %d   windowStride = %d\n", windowSize, windowStride);
      printf("   prob_original.l = %d\n", prob_original.l);
      printf("   I computed %d points not used at the end\n", numPointsNotUsedAtEnd);
      printf("   I computed lastPointUsedIndex = %d\n", lastPointUsedIndex);
      exit(0);
    }
  
  double *copy_y = Malloc(double,prob_original.l);
  memcpy(copy_y, prob_original.y, sizeof(double) * prob_original.l);

  struct feature_node **copy_x = Malloc(struct feature_node *,prob_original.l);
  memcpy(copy_x, prob_original.x, sizeof(struct feature_node *) * prob_original.l);

  int *copy_frameNumber_original = Malloc(int,prob_original.l);
  memcpy(copy_frameNumber_original, frameNumber_original, sizeof(int) * prob_original.l);

  /* now reverse all the y's, and
     reverse all the x pointers, and
     reverse the order of the frame numbers
  */
  for (int i = 0 ; i <= lastPointUsedIndex ; i++)
    {
      prob_original.y[i] = copy_y[lastPointUsedIndex - i];
      prob_original.x[i] = copy_x[lastPointUsedIndex - i];
      frameNumber_original[i] = copy_frameNumber_original[lastPointUsedIndex - i];
    }

  free(copy_y);
  free(copy_x);
  free(copy_frameNumber_original);
} // end: ReverseOriginalProblem()


/* firstLineNum is 0-based */
void CopyLines(struct problem *p_in, int in_firstLineNum, int numLinesToCopy,
               struct problem *p_out, int out_firstLineNum)
{
  //  printf("CopyLines(%d -> %d)  %d lines\n", in_firstLineNum,
  //         out_firstLineNum, numLinesToCopy);
  for (int i = 0 ; i < numLinesToCopy ; i++)
    {
      p_out->x[out_firstLineNum + i] = p_in->x[in_firstLineNum + i];
      p_out->y[out_firstLineNum + i] = p_in->y[in_firstLineNum + i];

      frameNumber_shuffled[out_firstLineNum + i] = frameNumber_original[in_firstLineNum + i];
      // printf("frameNumber_shuffled[%d] = frameNumber_original[%d]  ... %d\n",
      //        out_firstLineNum + i, in_firstLineNum + i,
      //        frameNumber_original[in_firstLineNum + i]);
    }
}


int myrandom (int i)
{
  return(std::rand()%i);
}



/* Memory for p_out is already allocated before this routine is called.
   This randomly moves blocks of data around in chunks.  The size of
   the chunks is defined by 'blockShuffleSize'.  Would be nice if that
   were an even multiple of p_in->l, but if not, the last few lines will
   never be moved. 
   Example: If there are 123 lines in the .train file, and blockShuffleSize
   is 10, lines 0..9 will be kept together, in order. Same with lines 10..19,
   20..29, ... 110.119.  Those 12 sets will be randomly shuffled.  The last 3
   lines (120, 121, 122) will never move.  They will always be at the end.
   (and depending on the windowSize / windowStride, may or may not be computed)
*/
void ShuffleProblem(struct problem *p_in, struct problem *p_out)
{
  p_out->l = p_in->l;
  p_out->n = p_in->n;
  p_out->bias = p_in->bias;

  std::vector<int> myvector;
  if (blockShuffleSize <= 0)
    {
      printf("WARNING: ShuffleProblem(): blockShuffleSize was %d. Setting to 1\n",
             blockShuffleSize);
      blockShuffleSize = 1;
    }
  
  int numBlocksToShuffle = prob_original.l / blockShuffleSize;
  if (numBlocksToShuffle == 0)
    {
      printf("WARNING: ShuffleSize: numBlocksToShuffle = %d\n", numBlocksToShuffle);
      printf("         Which means don't bother shuffling...so I won't\n");
      CopyProblem(p_in, p_out);
      return;  // RETURN
    }

  for (int i = 0; i < numBlocksToShuffle ; i++)
    myvector.push_back(i);  // 0, 1, ...
  std::random_shuffle ( myvector.begin(), myvector.end(), myrandom);
  //  std::random_shuffle ( myvector.begin(), myvector.end() );

  for (int i = 0; i < numBlocksToShuffle ; i++)
    {
      CopyLines(p_in, myvector[i] * blockShuffleSize, blockShuffleSize,
                p_out, i * blockShuffleSize);
    }
  /* And now copy all the remaining lines that don't get shuffled: */
  for (int i = numBlocksToShuffle * blockShuffleSize ; i < prob_original.l ; i++)
    {
      p_out->x[i] = p_in->x[i];
      p_out->y[i] = p_in->y[i];
      frameNumber_shuffled[i] = frameNumber_original[i];
    }
} // end: ShuffleProblem()


/* Memory for p_out is already allocated before this routine is called.
   We also copy the frame numbers.  Memory for those is already
   allocated as well.
*/
void CopyProblem(struct problem *p_in, struct problem *p_out)
{
  p_out->l = p_in->l;
  p_out->n = p_in->n;
  p_out->bias = p_in->bias;
  for (int i = 0 ; i < prob.l ; i++)
    {
      p_out->y[i] = p_in->y[i];
      p_out->x[i] = p_in->x[i];
      frameNumber_shuffled[i] = frameNumber_original[i];
    }
} // end: CopyProblem()

