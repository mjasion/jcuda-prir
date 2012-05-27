package pl.pw.edu.prir.tsole.howto;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;


/**
 * 
 * @author marek
 *
 *
 *Macierz przechowywana jest jako wektor  +  informacja o ilosci kolumn
 *
 */
public class Macierze {
	
	public static void main(String args[]){
		
		dodawanieMacierzy();
		
	}
	
	
	
	
	public static void dodawanieMacierzy(){
		System.out.println("dodawanie macierzy");
		
		int matrixSize = 5;
		//macierz A
		float A[] = new float[matrixSize*matrixSize];
		Pointer pa = new Pointer();
		
		//macierz C
		float C[] = new float[matrixSize*matrixSize];
		Pointer pc = new Pointer();
		
		
		for(int i =0; i < matrixSize * matrixSize; i++){
			A[i] = i;
			C[i] = 0;
		}
		
		 // wypisanie macierzy A
        System.out.println("A:");
        System.out.println(toString2D(A, matrixSize));
        
		
		
		//inicjalizacja
		JCublas.cublasInit();
		
		
		//alokacja pamieci
		int result =JCublas.cublasAlloc(matrixSize * matrixSize, Sizeof.FLOAT, pa);
		System.out.println("result alokacji = " + result);
		
		
		//przekopiowanie macierzy do gpu
		result = 13;
		result = JCublas.cublasSetMatrix(matrixSize, matrixSize, Sizeof.FLOAT, Pointer.to(A), matrixSize, pa, matrixSize);
//		result = JCublas.cublasSetVector(matrixSize * matrixSize, Sizeof.FLOAT, Pointer.to(A), 1, pa, 1);
		System.out.println("result przenoszenia = " + result);
		
		
		//pobranie macierzy z gpu do macierzy C
		JCublas.cublasGetMatrix(matrixSize, matrixSize, Sizeof.FLOAT, pa, matrixSize, Pointer.to(C), matrixSize);
		
//		JCublas.cublasGetVector(matrixSize* matrixSize, Sizeof.FLOAT, Pointer.to(pa), 1, C,1);
		
		JCublas.cublasFree(pa);
		
		 // wypisanie macierzy C
        System.out.println("C:");
        System.out.println(toString2D(C, matrixSize));
		
		
		
		
		//finito
		JCublas.cublasShutdown();
		
		
	}
	
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
