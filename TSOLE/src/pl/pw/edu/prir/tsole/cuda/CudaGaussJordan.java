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
 *         POST : funkcja zwalnia tylko zasoby przez siebie zaalokowane, srodowisko trzeba wylaczyc  w miejscu wywolania
 *         
 */
public class CudaGaussJordan implements IMatrixCompute {
	private static float[][] matA = {{0, 2, 1} , {2, 3, 1}, {1, 1, 4}};
	private static float[][] matB = {{11}, {13}, {12}};

	@Override
	public float[] computeMatrix(float[][] A, float[][] Y) {
		
		long start = System.nanoTime();
		long end,allocTime;
		
		int rows = A.length;
		int preCols = A[0].length;
		int cols = preCols + 1; // ilosc kolumn w macierzy polaczonej ( [A|Y] )
		
		float[] divideFactor = new float[1];
		float[] multiplyFactor = new float[1];
		float[] diagonal = new float[1];
		
		float[] resultMatrix = new float[rows*cols];
		float[] resultVector = new float[rows];
		

		float[] combinedVector = TsoleUtils.makeCombinedVectorFromMatrixes(A, rows, preCols, Y, 1, rows);
		Pointer pa = new Pointer();
		Pointer pj = new Pointer();

		//alokacja
		cublasAlloc(rows * cols, Sizeof.FLOAT, pa);
		cublasAlloc(cols, Sizeof.FLOAT, pj);
		//kopiowanie do gpu
		JCublas.cublasSetVector(rows * cols, Sizeof.FLOAT, Pointer.to(combinedVector), 1, pa, 1);
		allocTime = System.nanoTime();
	for(int i=0; i < rows; i++){
			
			/*pobranie przekatnej */
			JCublas.cublasGetVector(1, Sizeof.FLOAT, pa.withByteOffset( ((i*rows) + i)*Sizeof.FLOAT), 1, Pointer.to(diagonal), 1);
			
			if(diagonal[0] == 0){
				//1. znajdz indeks najwiekszego elementu w i-tej kolumnie
				int indexMax = JCublas.cublasIsamax(rows, pa.withByteOffset((i*rows) * Sizeof.FLOAT), 1);	//
				indexMax= indexMax-1;
				//2. zamien wiersze
				JCublas.cublasSswap(cols, pa.withByteOffset((i*rows) * Sizeof.FLOAT), rows, pa.withByteOffset((indexMax) * Sizeof.FLOAT), rows);
				
				//3. wez nowy wspolczynnik
				JCublas.cublasGetVector(1, Sizeof.FLOAT, pa.withByteOffset(((i * rows) + i) * Sizeof.FLOAT), 1, Pointer.to(diagonal), 1);
			}
			
			
			if(diagonal[0] != 0){
				//podziel wiersz [i] przez macierz[i][i]
				
				/*pobranie wspolczynnika prze ktory ma dzielic  (xinc jest nie wazny bo pobieram tylko 1 element !)*/
				JCublas.cublasGetVector(1, Sizeof.FLOAT, pa.withByteOffset( ((i*rows) + i)*Sizeof.FLOAT), 1, Pointer.to(divideFactor), 1);
				/* -----------------------------------------*/
				float wsp = 1;;
				if(divideFactor[0] != 0){
					wsp =(1/divideFactor[0]);
				}else{
					System.out.println("divide[0] = 0");
				}
				
				JCublas.cublasSscal(cols, wsp, pa.withByteOffset(i*Sizeof.FLOAT), rows);
				
				for(int j=0; j < preCols; j++){
					if(j != i){
						//kopiowanie i-tego wiersza
						JCublas.cublasScopy(cols, pa.withByteOffset(i*Sizeof.FLOAT), rows, pj, 1);
						/* pobranie wspolczynnika przez ktory ma dzielic A(j,i)  i  mnozenie i-tego wiersza przez A(j,i) */
						JCublas.cublasGetVector(1, Sizeof.FLOAT, pa.withByteOffset((i*rows + j)*Sizeof.FLOAT), 1, Pointer.to(multiplyFactor), 1);
						JCublas.cublasSscal(cols, multiplyFactor[0], pj, 1); 
						
						//odjecie Saxpyz
						JCublas.cublasSaxpy(cols, -1, pj, 1, pa.withByteOffset(j*Sizeof.FLOAT), rows);
						
					}
				}
				
			}
//			JCublas.printMatrix(cols, pa, rows);
		}
//	JCublas.printMatrix(cols, pa, rows);
	
	//pobranie macierzy wynikowej (A | X) z gpu
	JCublas.cublasGetVector(rows*cols, Sizeof.FLOAT, pa, 1, Pointer.to(resultMatrix), 1);
	
	JCublas.cublasFree(pj);
	JCublas.cublasFree(pa);
	
	end = System.nanoTime();
	System.out.println("\n[Cuda Gauss-Jordan] czas alokacji głównej macierzy(wektora) do gpu : " + (allocTime-start) + " ns.");
	System.out.println("[Cuda Gauss-Jordan] czas obliczeń  : " + (end-allocTime) + " ns.");
	System.out.println("[Cuda Gauss-Jordan] całkowity czas działania  : " + (end-start) + " ns.");
	
	return TsoleUtils.getResultsFromResultVector(resultMatrix, rows, cols);
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
//		 float[][] matrixA = IOLogic.readMatrix("matrix_simple_a");
//		 float[][] matrixB = IOLogic.readMatrix("matrix_simple_b");
		 float[][] matrixA = IOLogic.readMatrix("matrix1000_A");
		 float[][] matrixB = IOLogic.readMatrix("matrix1000_B");
		 
		JCublas.cublasInit();
//		JCublas.setLogLevel(LogLevel.LOG_TRACE);
		
		
		float[] result = new CudaGaussJordan().computeMatrix(matrixA, matrixB);
//		float[] result = new CudaGaussJordan().computeMatrix(matA, matB);
		
		System.out.println("\n\n**************** <RESULT> ********************");
		TsoleUtils.printMatrix(result);
		System.out.println("**************** </RESULT> ********************\n\n");
		JCublas.cublasShutdown();

	}

}
