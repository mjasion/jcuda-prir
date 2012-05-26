package pl.pw.edu.prir.tsole;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;


/**
 * 
 * 
 * SGEEM - to operacja mnozenia macierzy pojedynczej precyzji
 *
 */


public class marekTesting {
	
	/* Matrix size */
	private static final int N = 10;
	
	public static void main(String args[]){
		
		float h_A[];
	    float h_B[];
	    float h_C[];
	    Pointer d_A = new Pointer();
	    Pointer d_B = new Pointer();
	    Pointer d_C = new Pointer();
	    float alpha = 1.0f;
	    float beta = 0.0f;
	    int n2 = N * N;
	    int i;

	    /* Initialize JCublas */
	    JCublas.cublasInit();
	    System.out.println("inicjalizacja");
	    /* Allocate host memory for the matrices */
	    h_A = new float[n2];
	    h_B = new float[n2];
	    h_C = new float[n2];

	    /* Fill the matrices with test data */
	    for (i = 0; i < n2; i++)
	    {
	      h_A[i] = (float)Math.random();
	      h_B[i] = (float)Math.random();
	      h_C[i] = (float)Math.random();
	    }

	    /* Allocate device memory for the matrices */
	    //alokacja miejsca na gpu
	    JCublas.cublasAlloc(n2, Sizeof.FLOAT, d_A);
	    JCublas.cublasAlloc(n2, Sizeof.FLOAT, d_B);
	    JCublas.cublasAlloc(n2, Sizeof.FLOAT, d_C);

	    /* Initialize the device matrices with the host matrices */
	    //kopiowanie lokalnych wektorow do pamieci gpu
	    JCublas.cublasSetVector(n2, Sizeof.FLOAT, Pointer.to(h_A), 1, d_A, 1);
	    JCublas.cublasSetVector(n2, Sizeof.FLOAT, Pointer.to(h_B), 1, d_B, 1);
	    JCublas.cublasSetVector(n2, Sizeof.FLOAT, Pointer.to(h_C), 1, d_C, 1);

	    /**
	     * 
	     * 
	     * SGEEM - to operacja mnozenia macierzy pojedynczej precyzji
	     * 
	     * http://www.jcuda.org/jcuda/jcublas/doc/jcuda/jcublas/JCublas.html
	     * 
	     * C = alpha * op(A) * op(B) + beta * C, where op(X) is one of op(X) = X or op(X) = transpose(X)
	     */
	    
	    /* Performs operation using JCublas */
	    JCublas.cublasSgemm('n', 'n', N, N, N, alpha, d_A, N, d_B, N, beta, d_C, N);

	    /* Read the result back */
	    //zapisanie z gpu do lokalnej pamieci
	    JCublas.cublasGetVector(n2, Sizeof.FLOAT, d_C, 1, Pointer.to(h_C), 1);

	    /* Memory clean up */
	    JCublas.cublasFree(d_A);
	    JCublas.cublasFree(d_B);
	    JCublas.cublasFree(d_C);

	    System.out.println("zwalnianie");
	    /* Shutdown */
	    JCublas.cublasShutdown();
	
	}

}
