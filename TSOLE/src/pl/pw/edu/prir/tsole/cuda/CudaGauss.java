package pl.pw.edu.prir.tsole.cuda;

import static jcuda.jcublas.JCublas.cublasAlloc;
import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.io.IOLogic;

/**
 * 
 * @author marek
 * 
 *         Obliczenie liniowego ukladu rownan wedlug algorytmu Gaussa-Jordana z
 *         uzyciem JCudy a dokladniej biblioteki JCublas
 * 
 * 
 *         PRE : srodowisko JCublas musi byc zainicjowane przed wywolaniem
 *         funkcji computeMatrix !!
 * 
 *         POST : funkcja zwalnia tylko zasoby przez siebie zaalokowane,
 *         srodowisko trzeba wylaczyc w miejscu wywolania
 * 
 */
public class CudaGauss implements IMatrixCompute {

	@Override
	public float[] computeMatrix(float[][] A, float[][] Y) {

		int rows = A.length;
		int preCols = A[0].length;
		int cols = preCols + 1; // ilosc kolumn w macierzy polaczonej ( [A|Y] )

		float[] yFactor = new float[1];
		float[] multiplyFactor = new float[1];
		float[] diagonal = new float[1];

		float[] resultMatrix = new float[rows * cols];
		float[] yVector = new float[rows];

		float[] combinedVector = TsoleUtils.makeCombinedVectorFromMatrixes(A, rows, preCols, Y, 1, rows);
		Pointer pa = new Pointer();
		Pointer pi = new Pointer();
		
		
		// alokacja
		cublasAlloc(rows * cols, Sizeof.FLOAT, pa);
		cublasAlloc( cols, Sizeof.FLOAT, pi);
		// kopiowanie do gpu
		JCublas.cublasSetVector(rows * cols, Sizeof.FLOAT, Pointer.to(combinedVector), 1, pa, 1);

		// algorytm
		
		// 1) Eliminacja zmiennych
		JCublas.printMatrix(cols, pa, rows);
		
		for (int i = 0; i < rows; i++) {
			// pobierz wspolczynnik matrix[i][i]
			JCublas.cublasGetVector(1, Sizeof.FLOAT, pa.withByteOffset(((i * rows) + i) * Sizeof.FLOAT), 1, Pointer.to(diagonal), 1);

			// podziel wiersz [i] przez diagonal
			float wsp = (float) (1 / diagonal[0]);
//			JCublas.printVector(1, pa.withByteOffset(i * Sizeof.FLOAT));	//wypisanie odwrotnosci wsp diag
			JCublas.cublasSscal(cols, wsp, pa.withByteOffset(i * Sizeof.FLOAT), rows);
			
			for (int j = i + 1; j < rows; j++) {
				// pobierz wspolczynnik matrix[j][i]
				JCublas.cublasGetVector(1, Sizeof.FLOAT, pa.withByteOffset(((i * rows) + j) * Sizeof.FLOAT), 1, Pointer.to(multiplyFactor), 1);
//				JCublas.printVector(1, pa.withByteOffset(((i * rows) + i) * Sizeof.FLOAT));	//wypisanie odwrotnosci wsp diag
				/* [j] wiersz　= [j] wiersz - ( [i] wiersz * wspo ) */
				
				JCublas.cublasScopy(cols, pa.withByteOffset(i*Sizeof.FLOAT), rows, pi, 1);	//kopiujemy [i] wiersz to tempa
				
				multiplyFactor[0] = -multiplyFactor[0]; // aby otrzymac odejmowanie
				JCublas.cublasSaxpy(cols, multiplyFactor[0], pi, 1, pa.withByteOffset(j * Sizeof.FLOAT), rows); // !
			}
//			JCublas.printMatrix(cols, pa, rows);

		}
		JCublas.printMatrix(cols, pa, rows);

		// 2) waskie gardlo, czesc sekwencyjna - Podstawianie odwrotne

		/* Y po 1 czesci
		 * 4x1 [2.269565] [0.031079847] [-1.1226826] [10.185098]
		 */
		
		//pobranie macierzy 
		JCublas.cublasGetVector(rows*cols, Sizeof.FLOAT, pa, 1, Pointer.to(resultMatrix), 1);
		
		float[][] matrixAfterFirstStage = TsoleUtils.vectorToMAtrix(resultMatrix, rows, cols);
		yVector = TsoleUtils.getResultsFromResultVector(resultMatrix, rows, cols);
		System.out.println("***************RESULT VECTOR*******************");
		float results[] = new float[rows];
		
		for(int i=rows-1; i>=0; i--) {
			float s = yVector[i];
			System.out.println(s);
			
			for(int j=preCols-1; j>i; j--) {
				s =s - (matrixAfterFirstStage[i][j] * results[j]);
			}
			results[i] = s;
		}
		
		/*
		 * [-8.154252] [-0.26902813] [-2.8261123] [10.185095]
		 */
		
		JCublas.cublasFree(pi);
		JCublas.cublasFree(pa);

		return results;
	}

	/**
	 * testowanie
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		PropertyConfigurator.configure("log4j.properties");
//		float[][] matrixA = IOLogic.readMatrix("matrixA");
//		float[][] matrixB = IOLogic.readMatrix("matrixB");
		 float[][] matrixA = IOLogic.readMatrix("matrix_simple_a");
		 float[][] matrixB = IOLogic.readMatrix("matrix_simple_b");
		JCublas.cublasInit();
		JCublas.setLogLevel(LogLevel.LOG_TRACE);

		float[] result = new CudaGauss().computeMatrix(matrixA, matrixB);
		System.out.println("\n\n**************** <RESULT> ********************");
		TsoleUtils.printMatrix(result);
		System.out.println("**************** </RESULT> ********************\n\n");
		JCublas.cublasShutdown();

	}

}
