package pl.pw.edu.prir.tsole;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.howto.Macierze;
import pl.pw.edu.prir.tsole.io.IOLogic;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;

/**
 * Parametry VM: -Djava.library.path=${workspace_loc:TSOLE/JCuda-0.4.1/lib}
 * 
 * @author Marcin, marek
 * 
 */

public class Main {

	public static void main(String[] args) {
		PropertyConfigurator.configure("log4j.properties");

		float[][] matrix = IOLogic.readMatrix("matrix44");
		float[][] matrix2 = IOLogic.readMatrix("matrix44_2");
		int rows = matrix.length;
		int cols = matrix[0].length;
		IOLogic.printMatrix(matrix);
		IOLogic.printMatrix(matrix2);

		/* do jcudy */
		// matrix A
		Pointer pa = new Pointer();
		float cudaVector_A[] = Macierze.matrixToVector(matrix, rows, matrix[0].length);
		// matrix B
		Pointer pb = new Pointer();
		float cudaVector_B[];
		// matrix C
		Pointer pc = new Pointer();
		float cudaVector_C[] = new float[rows * cols];
		float resultMatrix_C[][];

		// inicjalizacja jcublas
		JCublas.cublasInit();
		
		// alokacja pamieci
		int result = JCublas.cublasAlloc(rows * cols, Sizeof.FLOAT, pa);
		
		// przerzuceie danych do gpu
		JCublas.cublasSetMatrix(rows, cols, Sizeof.FLOAT, Pointer.to(cudaVector_A), rows, pa, cols);
		
		
		
		// wywolanie funkcji ,ktora beda obliczac w cudzie
		
		
		//pobranie wynikow
		JCublas.cublasGetMatrix(rows, cols, Sizeof.FLOAT, pa, rows, Pointer.to(cudaVector_C), cols);
		
		System.out.println("pobrane z gpu<< powinno sie zgadzac z 1 macierza>>");
		resultMatrix_C = Macierze.vectorToMAtrix(cudaVector_C, rows, cols);
		IOLogic.printMatrix(resultMatrix_C);
		
		//zwolnienie pamieci
		JCublas.cublasFree(pa);
		
		//
		JCublas.cublasShutdown();

		// Float matrixAfter[][] = Macierze.vectorToMAtrix(cudaVector_A,
		// matrix.length, matrix[0].length);

	}

}
