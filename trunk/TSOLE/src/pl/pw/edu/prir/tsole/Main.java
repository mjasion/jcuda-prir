package pl.pw.edu.prir.tsole;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.howto.Macierze;
import pl.pw.edu.prir.tsole.io.IOLogic;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

/**
 * Parametry VM: 
 * -Djava.library.path=${workspace_loc:TSOLE/JCuda-0.4.1/lib}
 * @author Marcin
 *
 */

public class Main {

	public static void main(String[] args) {
		PropertyConfigurator.configure("log4j.properties");
		
		Float[][] matrix = IOLogic.readMatrix("matrix44");
		Float[][] matrix2 = IOLogic.readMatrix("matrix44_2");
		IOLogic.printMatrix(matrix);
		IOLogic.printMatrix(matrix2);
		
		
		
		
	}

}
