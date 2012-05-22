package pl.pw.edu.prir.tsole;

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
		Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: "+pointer);
        JCuda.cudaFree(pointer);
	}

}
