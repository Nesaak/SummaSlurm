# Example Makefile. Before compile,
#module load openmpi/5.0.1
CC = mpicc
CLINKER = mpicc
F77 = mpif77
CFLAGS = -O3
FFLAGS = -O3
MATH_LIB = -lm

summa: summa.c
	$(CLINKER) -o summa summa.c $(MATH_LIB)
.c.o:
	$(CC) $(CFLAGS) -c $<
.f.o:
	$(F77) $(FFLAGS) -c $<
clean:
	rm -f *.o summa

