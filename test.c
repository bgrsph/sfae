#include "spkernels.h"  /** TODO: Fix this in the cluster **/

int main(int argc, char *argv[])
{
   /*----------------------------------- *
    *  Driver program for GPU L/U solve  *
    *  x = U^{-1} * L^{-1} * b           *
    *  Kernels provided:                 *
    *  CPU L/U solve                     *
    *  GPU L/U solve w/ level-scheduling *
    *  GPU L/U solve w/ sync-free        *
    *----------------------------------- */
   int i, n, nnz, nx = 32, ny = 32, nz = 32, npts = 7, flg = 0, mm = 1, dotest = 0;
   HYPRE_Real *h_b, *h_x0, *h_x1, *h_x2, *h_x3, *h_x4, *h_x5, *h_x6, *h_x7, *h_x8, *h_z;
   struct coo_t h_coo;
   hypre_CSRMatrix *h_csr;
   double e1, e2, e3, e4, e5, e6, e7, e8;
   char fname[2048];
   int NTESTS = 10;
   int REP = 10;
   double err;

   /*-----------------------------------------*/
   flg = findarg("help", NA, NULL, argc, argv);
   if (flg)
   {
      printf("Usage: ./testL.ex -nx [int] -ny [int] -nz [int] -npts [int] -mat fname -mm [int] -rep [int] -dotest\n");
      return 0;
   }
   //srand (SEED);
   //srand(time(NULL));
   /*---------- Init GPU */
   cuda_init(argc, argv);
   /*---------- cmd line arg */
   findarg("nx", INT, &nx, argc, argv);
   findarg("ny", INT, &ny, argc, argv);
   findarg("nz", INT, &nz, argc, argv);
   findarg("npts", INT, &npts, argc, argv);
   flg = findarg("mat", STR, fname, argc, argv);
   findarg("mm", INT, &mm, argc, argv);
   findarg("rep", INT, &REP, argc, argv);
   dotest = findarg("dotest", NA, &dotest, argc, argv);

   /** BUGRA
    * example_matrix_list.txt file has one matrix name (without .mtx extension) in each line. 
    * For example, ash331.mtx file has to be in the same folder as test.c
    * TODO: I can use wget to install the mtx files into some other folder in order to be systematic
    **/

   FILE *file_stream;
   char *matrix_name = NULL;
   size_t matrix_name_length = 0;
   size_t read;
   file_stream = fopen("example_matrix_list.txt", "r");
   if (file_stream == NULL)
   {
      exit(EXIT_FAILURE);
   }
   while ((read = getline(&matrix_name, &matrix_name_length, file_stream)) != -1)
   {
      matrix_name[strlen(matrix_name) - 1] = 0;
      strcat(matrix_name, ".mtx");
      strcpy(fname, matrix_name);
      printf("Starting to process matrix: %s\n", fname);

      /*---------- Read from Martrix Market file */
      if (flg == 1)
      {
         read_coo_MM(fname, mm, 0, &h_coo);
      }
      else
      {
         lapgen(nx, ny, nz, &h_coo, npts);
      }
      n = h_coo.nrows;
      nnz = h_coo.nnz;
      /*---------- COO -> CSR */
      coo_to_csr(0, &h_coo, &h_csr);
      /*--------------------- vector b */
      h_b = (HYPRE_Real *)malloc(n * sizeof(HYPRE_Real));
      h_z = (HYPRE_Real *)malloc(n * sizeof(HYPRE_Real));
      for (i = 0; i < n; i++)
      {
         //h_b[i] = rand() / (RAND_MAX + 1.0);
         //h_z[i] = rand() / (RAND_MAX + 1.0);
         h_b[i] = cos(i + 1);
         h_z[i] = sin(i + 1);
      }
      if (!dotest)
      {
         goto bench;
      }
      /*------------------------------------------------ */
      /*------------- Start testing kernels ------------ */
      /*------------------------------------------------ */
      printf("Test kernels for %d times ...\n", NTESTS);

      /*------------ GPU L/U Solv w/ Col Dyn-Sched */
      err = 0.0;
      h_x5 = (HYPRE_Real *)malloc(n * sizeof(HYPRE_Real));
      printf(" [GPU] G-S DYNC,       ");
      for (int i = 0; i < NTESTS; i++)
      {
         memcpy(h_x5, h_z, n * sizeof(HYPRE_Real));
         GaussSeidelColDynSchd<true>(h_csr, h_b, h_x5, 1, false);
         e5 = error_norm(h_x0, h_x5, n);
         err = max(e5, err);
      }
      printf("err norm %.2e\n", err);
      free(h_x5);

   bench:
      printf("Benchmark kernels, repetition %d\n", REP);
      /*------------------------------------------------ */
      /*------------- Start benchmarking kernels ------- */
      /*------------------------------------------------ */

      /*------------ GPU L/U Solv w/ Dyn-Sched C */
      h_x5 = (HYPRE_Real *)malloc(n * sizeof(HYPRE_Real));
      memcpy(h_x5, h_z, n * sizeof(HYPRE_Real));
      GaussSeidelColDynSchd<false>(h_csr, h_b, h_x5, REP, true);
      e5 = error_norm(h_x0, h_x5, n);
      printf("err norm %.2e\n", e5);
      free(h_x5);

      /*----------- Done free */
      hypre_CSRMatrixDestroy(h_csr);
      FreeCOO(&h_coo);
      free(h_b);
      free(h_x0);
      /*---------- check error */
      cuda_check_err();

      //BUGRA: free the variable before the next iteration
      free(matrix_name);
      //BUGRA: close the file to be read in the next iteration
      fclose(file_stream);
      printf("\n\n");
   }
}