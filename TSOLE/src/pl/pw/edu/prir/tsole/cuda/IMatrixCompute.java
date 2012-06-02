package pl.pw.edu.prir.tsole.cuda;

/**
 * 
 * @author marek
 *
 *Interfejs implementowany przez kazda klase liczaca liniowy uklad rownan w postaci AX=Y
 *
 */


public interface IMatrixCompute {
	/**
	 * 
	 * @param 
	 * 			matrix macierz wspolczynnikow- A
	 * @param vector
	 * 			wektor wynikow - Y
	 * @return
	 * 		float[] - wektor rozwiazan - X
	 */
	public float[] computeMatrix(float[][] A, float[][] Y);

}
