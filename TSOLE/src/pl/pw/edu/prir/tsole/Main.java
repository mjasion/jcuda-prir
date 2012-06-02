package pl.pw.edu.prir.tsole;

import static jcuda.jcublas.JCublas.cublasAlloc;
import static jcuda.jcublas.JCublas.cublasSetMatrix;
import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.cuda.TsoleCuda;
import pl.pw.edu.prir.tsole.io.IOLogic;

/**
 * Parametry VM: -Djava.library.path=${workspace_loc:TSOLE/JCuda-0.4.1/lib}
 * 
 * @author Marcin, marek
 * 
 */

public class Main {

	private static String pathToMatrixA = null;
	private static String pathToMatrixB = null;
	private static String pathToOutputFile = null;
	private static String methodSelected = null;

	public static void main(String[] args) {
		PropertyConfigurator.configure("log4j.properties");

		if (Manual.parseArgs(args) == 1)
			return;

		Manual.printWelcome();

		float[][] matrix = IOLogic.readMatrix(pathToMatrixA);
		float[][] matrix2 = IOLogic.readMatrix(pathToMatrixB);
		int rows = matrix.length;
		int cols = matrix[0].length;

		IOLogic.printMatrix(matrix);
		IOLogic.printMatrix(matrix2);

		/* do jcudy */
		//macierz(wektor) A
//		float cudaVector_A[] = TsoleCuda.matrixToVector(matrix, rows, cols);
//		// odwrotnosc A
//		float invA[] = cudaVector_A.clone();
//		// wektor y(B)
//		float cudaVector_B[] = TsoleCuda.matrixToVector(matrix2, 1, rows);
//		// macierz(wektor) X
//		float cudaVector_C[] = new float[rows];
//
//		for (int i = 0; i < rows; i++)
//			cudaVector_C[i] = 0;

		// inicjalizacja jcublas
		JCublas.cublasInit();
		JCublas.setLogLevel(LogLevel.LOG_TRACE);


//		TsoleCuda.invertMatrix(rows, invA);
//		float jednostkowa[] = new float[rows * rows];
//		TsoleCuda.multiply(rows, cudaVector_A, invA, jednostkowa);

		float[] combinedVector = TsoleCuda.makeCombinedVectorFromMatrixes(matrix, rows, cols, matrix2, 1, rows);
		float[][] baseMatrix = TsoleCuda.vectorToMAtrix(combinedVector, rows, cols+1);
		
		System.out.println("base Matrix");
		IOLogic.printMatrix(baseMatrix);
		
		
		Pointer pa = new Pointer();
		
		cublasAlloc(rows*(cols+1), Sizeof.FLOAT, pa);
//		cublasSetMatrix(rows, cols+1, Sizeof.FLOAT, Pointer.to(combinedVector), cols+1, pa, cols+1);
		JCublas.cublasSetVector(rows*(cols+1), Sizeof.FLOAT, Pointer.to(combinedVector), 1, pa, 1);
		
//		JCublas.printMatrix(cols+1, pa, rows);
		
		//probowa przekopiowania wektora z macierzy
//		Pointer pw = new Pointer();
//		cublasAlloc(cols+1, Sizeof.FLOAT, pw);
//		
//		JCublas.cublasScopy(cols+1, pa, rows, pw, 1);
//		
//		JCublas.printVector(cols+1, pw);
		
		//podzielenie przez skalar
//		float alpha = (float)(1/matrix[0][3]);
//		System.out.println("alpha = " + alpha);
//		JCublas.cublasSscal(cols+1,alpha , pw, 1);
//		JCublas.printVector(cols+1, pw);
		
		
		/*
		 * 4x1
[-8.154254]
[-0.26902807]
[-2.8261123]
[10.185095]



		 */
		
		//zerowy
		float[] zerowy = new float[cols+1];
		
		for(float it: zerowy)
			it = 0f;
		
		/* pointera mozna wyciagnac na gore i alokowac tylko 1, po petli go zwalniac.
		 * 
		 */
		//GAUSS-JORDAN
		
		JCublas.printMatrix(cols+1, pa, rows);
		
		for(int i=0; i < rows; i++){
			
			/*pobranie przekatnej */
			float[] przekatna = new float[1];
			JCublas.cublasGetVector(1, Sizeof.FLOAT, pa.withByteOffset( ((i*rows) + i)*Sizeof.FLOAT), 1, Pointer.to(przekatna), 1);
			if(przekatna[0] != 0){
				//podziel wiersz i przez macierz[i][i]
//				Pointer pi = new Pointer();
//				cublasAlloc(cols+1, Sizeof.FLOAT, pi);
//				JCublas.cublasScopy(cols+1, pa.withByteOffset(i*Sizeof.FLOAT), rows, pi, 1);
				
				/*pobranie wspolczynnika prze ktory ma dzielic  (xinc jest nie wazny bo pobieram tylko 1 !)*/
				float[] wspTabl = new float[1];
				JCublas.cublasGetVector(1, Sizeof.FLOAT, pa.withByteOffset( ((i*rows) + i)*Sizeof.FLOAT), 1, Pointer.to(wspTabl), 1);
				/* -----------------------------------------*/
				float wsp = (float)(1/wspTabl[0]);
				JCublas.cublasSscal(cols+1, wsp, pa.withByteOffset(i*Sizeof.FLOAT), rows);
				
//				JCublas.cublasSscal(cols+1, (1/baseMatrix[i][i]), pa.withByteOffset(i*Sizeof.FLOAT), rows);
				
				/*testing*/
				Pointer ptesting =new Pointer();
				cublasAlloc(cols+1, Sizeof.FLOAT, ptesting);
				JCublas.cublasScopy(cols+1, pa.withByteOffset(i*Sizeof.FLOAT), rows, ptesting, 1);
				JCublas.printVector(cols+1, ptesting);
				/* --------- */
				
//				JCublas.cublasSscal(cols+1,(1/matrix[i][i]) , pi, 1);
				for(int j=0; j < cols; j++){
					if(j != i){
						Pointer pj = new Pointer();
						cublasAlloc(cols+1, Sizeof.FLOAT, pj);
						//kopiowanie i-tego wiersza
						JCublas.cublasSetVector(cols+1, Sizeof.FLOAT, Pointer.to(zerowy), 1, pj, 1);
						JCublas.cublasScopy(cols+1, pa.withByteOffset(i*Sizeof.FLOAT), rows, pj, 1);
						//mnozenie i-tego wiersza przez A(j,i)
							/* pobranie wspolczynnika przez ktory ma dzielic A(j,i) */
						float[] innerWspTabl = new float[1];
						JCublas.cublasGetVector(1, Sizeof.FLOAT, pa.withByteOffset((i*rows + j)*Sizeof.FLOAT), 1, Pointer.to(innerWspTabl), 1);
						JCublas.printVector(1, pa.withByteOffset(((i*rows) + j)*Sizeof.FLOAT));
						float innerWsp = innerWspTabl[0];
						
						JCublas.cublasSscal(cols+1, innerWsp, pj, 1); 
						
//						JCublas.cublasSscal(cols+1, baseMatrix[j][i], pj, 1); //!
						
						JCublas.printVector(cols+1, pj);
						//odjecie Saxpy
						JCublas.cublasSscal(cols+1, -1, pj, 1);
						JCublas.cublasSaxpy(cols+1, 1, pj, 1, pa.withByteOffset(j*Sizeof.FLOAT), rows);
						
						/*testing */
						Pointer ptest = new Pointer();
						cublasAlloc(cols+1, Sizeof.FLOAT, ptest);
						JCublas.cublasScopy(cols+1, pa.withByteOffset(j*Sizeof.FLOAT), rows, ptest, 1);
						JCublas.printVector(cols+1, ptest);
						JCublas.cublasFree(ptest);
						
						/*--------*/
						JCublas.cublasFree(pj);
					}
				}
				
			}
			JCublas.printMatrix(cols+1, pa, rows);
		}
		
		JCublas.printMatrix(cols+1, pa, rows);
		
		
		float[] givenVector = new float[rows*(cols+1)];
		JCublas.cublasGetVector(rows*(cols+1), Sizeof.FLOAT, pa, 1, Pointer.to(givenVector), 1);
//		JCublas.cublasGetMatrix(rows, cols+1, Sizeof.FLOAT, pa, rows, Pointer.to(givenVector), rows);
		
		System.out.println("****************** given matrix from cuda ********************");
		float[][] givenMatrix = TsoleCuda.vectorToMAtrix(givenVector, rows, cols+1);
		IOLogic.printMatrix(givenMatrix);
////		System.out.println("combined vector");
////		
////		for(float it: combinedVector)
////			System.out.println(it);
//		
//		float combinedMatrix[][] = TsoleCuda.vectorToMAtrix(combinedVector, rows, cols+1);
//
//		System.out.println("given matrix");
//		IOLogic.printMatrix(combinedMatrix);
//		for(int i =0; i < rows; i++){
//			for(int j = 0; j < cols +1; j++)
//				System.out.print(combinedMatrix[i][j] + "   ");
//			System.out.println();
//		}
//		System.out.println();
////		System.out.println("macierz jednostkowa");
////		IOLogic.printMatrix(TsoleCuda.vectorToMAtrix(jednostkowa, rows, cols));
//		System.out.println("\nmacierz A");
//		IOLogic.printMatrix(TsoleCuda.vectorToMAtrix(cudaVector_A, rows, cols));
////		System.out.println("\nmacierz inv A");
////		IOLogic.printMatrix(TsoleCuda.vectorToMAtrix(invA, rows, cols));
//		System.out.println("\nwektor X");
//		TsoleCuda.multiply(rows, invA, cudaVector_B, cudaVector_C);
//		IOLogic.printMatrix(cudaVector_C);
		
		JCublas.cublasFree(pa);
		
//		JCublas.cublasFree(pw);
		
		JCublas.cublasShutdown();
	}

	public static String getPathToMatrixA() {
		return pathToMatrixA;
	}

	public static String getPathToMatrixB() {
		return pathToMatrixB;
	}

	public static String getMethodSelected() {
		return methodSelected;
	}

	public static void setPathToMatrixA(String pathToMatrixA) {
		Main.pathToMatrixA = pathToMatrixA;
	}

	public static void setPathToMatrixB(String pathToMatrixB) {
		Main.pathToMatrixB = pathToMatrixB;
	}

	public static void setMethodSelected(String methodSelected) {
		Main.methodSelected = methodSelected;
	}

	public static String getPathToOutputFile() {
		return pathToOutputFile;
	}

	public static void setPathToOutputFile(String pathToOutputFile) {
		Main.pathToOutputFile = pathToOutputFile;
	}

}
