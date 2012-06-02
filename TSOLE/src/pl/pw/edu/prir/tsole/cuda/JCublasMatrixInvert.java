package pl.pw.edu.prir.tsole.cuda;
/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * http://www.jcuda.org
 * 
 * Copyright 2011 Marco Hutter - http://www.jcuda.org
 * 
 * This example is based on a post in the JCuda forum at
 * http://forum.byte-welt.de/forumdisplay.php?f=87&langid=2
 */

import static jcuda.jcublas.JCublas.*;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

import java.util.Random;

import jcuda.*;

/**
 * Example of a matrix inversion using JCublas.
 */
public class JCublasMatrixInvert
{
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        cublasInit();

        // Create the input matrix
        int size = 7;
        float A[] = createRandomFloatData(size * size);

        // Invert the matrix
        float invA[] = A.clone();
        invertMatrix(size, invA);

        // Compute A*invA, which should yield the identity matrix
        float identity[] = new float[size * size];
        multiply(size, A, invA, identity);

        // Print the results
        System.out.println("A:");
        System.out.println(toString2D(A, size));
        System.out.println("invA:");
        System.out.println(toString2D(invA, size));
        System.out.println("identity:");
        System.out.println(toString2D(identity, size));

        cublasShutdown();
    }

    /**
     * Inverts the given n x n matrix. This method will initialize CUBLAS by
     * calling cublasInit, invert the matrix using
     * {@link #invertMatrix(int, float[])}, and shut down CUBLAS by calling
     * cublasShutdown.
     * 
     * @param n The size of the matrix
     * @param A The matrix.
     */
    public static void cublasInvertMatrix(int n, float A[])
    {
        cublasInit();
        invertMatrix(n, A);
        cublasShutdown();
    }

    /**
     * Copies the given n x n matrix into device memory, inverts it by calling
     * {@link #invertMatrix(int, Pointer)}, and copies it back into the given
     * array. <br />
     * <br />
     * This method assumes that CUBLAS already has been initialized by calling
     * cublasInit.
     * 
     * @param n The size of the matrix
     * @param A The matrix
     */
    public static void invertMatrix(int n, float A[])
    {
        Pointer dA = new Pointer();
        cublasAlloc(n * n, Sizeof.FLOAT, dA);
        cublasSetMatrix(n, n, Sizeof.FLOAT, Pointer.to(A), n, dA, n);

        invertMatrix(n, dA);

        cublasGetMatrix(n, n, Sizeof.FLOAT, dA, n, Pointer.to(A), n);
        cublasFree(dA);
    }

    /**
     * Invert the n x n matrix that is given in device memory.<br />
     * <br />
     * This method assumes that CUBLAS already has been initialized by calling
     * cublasInit.
     * 
     * @param n The size of the matrix
     * @param dA The matrix
     */
    public static void invertMatrix(int n, Pointer dA)
    {
        // Perform LU factorization
        int[] pivots = cudaSgetrfSquare(n, dA);

        // Perform inversion on factorized matrix
        cudaSgetri(n, dA, pivots);
    }

    /**
     * Convenience method that returns a pointer with the given offset (in
     * number of 4-byte float elements) from the given pointer.
     * 
     * @param p The pointer
     * @param floatOffset The offset
     * @return The new pointer
     */
    private static Pointer at(Pointer p, int floatOffset)
    {
        return p.withByteOffset(floatOffset * Sizeof.FLOAT);
    }

    /**
     * cudaSgetrf performs an in-place LU factorization on a square matrix. Uses
     * the unblocked BLAS2 approach
     * 
     * @param n The matrix size
     * @param dA The pointer to the matrix (in device memory)
     * @return The pivots
     */
    private static int[] cudaSgetrfSquare(int n, Pointer dA)
    {
        int[] pivots = new int[n];
        for (int i = 0; i < n; i++)
        {
            pivots[i] = i;
        }

        float[] factor = { 0.0f };
        Pointer pFactor = Pointer.to(factor);
        for (int i = 0; i < n - 1; i++)
        {
            Pointer offset = at(dA, i * n + i);

            int pivot = i - 1 + cublasIsamax(n - i, offset, 1);
            if (pivot != i)
            {
                pivots[i] = pivot;
                cublasSswap(n, at(dA, pivot), n, at(dA, i), n);
            }

            cublasGetVector(1, Sizeof.FLOAT, offset, 1, pFactor, 1);
            cublasSscal(n - i - 1, 1 / factor[0], at(offset, 1), 1);
            cublasSger(n - i - 1, n - i - 1, -1.0f, 
                at(offset, 1), 1, at(offset, n), n, at(offset, n + 1), n);
        }
        return pivots;
    }

    /***
     * cudaSgetri Computes the inverse of an LU-factorized square matrix
     * 
     * @param n
     *            The matrix size
     * @param dA
     *            The matrix in device memory
     * @param pivots
     *            The pivots
     */
    private static void cudaSgetri(int n, Pointer dA, int[] pivots)
    {
        // Perform inv(U)
        cudaStrtri(n, dA);

        // Solve inv(A)*L = inv(U)
        Pointer dWork = new Pointer();
        cublasAlloc(n - 1, Sizeof.FLOAT, dWork);

        for (int i = n - 1; i > 0; i--)
        {
            Pointer offset = at(dA, ((i - 1) * n + i));
            cudaMemcpy(dWork, offset, (n - 1) * Sizeof.FLOAT, 
                cudaMemcpyDeviceToDevice);
            cublasSscal(n - i, 0, offset, 1);
            cublasSgemv('n', n, n - i, -1.0f, 
                at(dA, i * n), n, dWork, 1, 1.0f, at(dA, ((i - 1) * n)), 1);
        }

        cublasFree(dWork);

        // Pivot back to original order
        for (int i = n - 1; i >= 0; i--)
        {
            if (i != pivots[i])
            {
                cublasSswap(n, at(dA, i * n), 1, at(dA, pivots[i] * n), 1);
            }
        }

    }

    /***
     * cudaStrtri Computes the inverse of an upper triangular matrix in place
     * Uses the unblocked BLAS2 approach
     * 
     * @param n The size of the matrix
     * @param dA The matrix
     */
    private static void cudaStrtri(int n, Pointer dA)
    {
        float[] factor = { 0.0f };
        Pointer pFactor = Pointer.to(factor);
        for (int i = 0; i < n; i++)
        {
            Pointer offset = at(dA, i * n);
            cublasGetVector(1, Sizeof.FLOAT, at(offset, i), 1, pFactor, 1);
            factor[0] = 1 / factor[0];
            cublasSetVector(1, Sizeof.FLOAT, pFactor, 1, at(offset, i), 1);
            cublasStrmv('u', 'n', 'n', i, dA, n, offset, 1);
            cublasSscal(i, -factor[0], offset, 1);
        }
    }

    // === Utility methods for this sample ====================================

    /**
     * Creates a new array with the given size, containing random data
     * 
     * @param size The size of the array
     * @return The array
     */
    private static float[] createRandomFloatData(int size)
    {
        Random random = new Random(0);
        float a[] = new float[size];
        for (int i = 0; i < size; i++)
        {
            a[i] = random.nextFloat();
        }
        return a;
    }

    /**
     * Multiplies the matrices A and B and writes the result into C.
     * 
     * @param size The size of the matrices
     * @param A Matrix A
     * @param B Matrix B
     * @param C Matrix C
     */
    private static void multiply(int size, float A[], float B[], float C[])
    {
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();

        cublasAlloc(size * size, Sizeof.FLOAT, dA);
        cublasAlloc(size * size, Sizeof.FLOAT, dB);
        cublasAlloc(size * size, Sizeof.FLOAT, dC);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);

        cublasSgemm('n', 'n', size, size, size, 1, 
            dA, size, dB, size, 0, dC, size);

        cublasGetVector(size * size, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
        cublasFree(dA);
        cublasFree(dB);
        cublasFree(dC);
    }

    /**
     * Creates a string representation of the given array as a matrix with with
     * given number of columns.
     * 
     * @param a The array
     * @param columns The number of columns
     * @return The string representation
     */
    private static String toString2D(float[] a, int columns)
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++)
        {
            if (i > 0 && i % columns == 0)
            {
                sb.append("\n");
            }
            sb.append(String.format("%7.4f ", a[i]));
        }
        return sb.toString();
    }

}