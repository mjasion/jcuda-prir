package pl.pw.edu.prir.tsole;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println(args[0]);
		Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: "+pointer);
        JCuda.cudaFree(pointer);
	}

}
