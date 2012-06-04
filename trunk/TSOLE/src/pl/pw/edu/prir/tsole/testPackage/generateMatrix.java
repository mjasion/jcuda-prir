package pl.pw.edu.prir.tsole.testPackage;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * 
 * @author marek
 *
 */
public class generateMatrix {
	
	public static void main(String[] args) throws IOException{
		int size = 10000;
		int size2= 1;
		Random r = new Random();
		float f;
		int multiply;
		String s;
		FileWriter fw = new FileWriter("matrix10000_B");
		fw.write(size + "x"+size2+"\n");
		for(int i =0; i < size; i++){
			for(int j=0; j < size2; j++){
				f = r.nextFloat();
				multiply = r.nextInt(10);
				f *= multiply;
				s = String.format("%2.1f", f);
				fw.write(s + " ");
			}
			fw.write("\n");
		}
		
		
		fw.close();
		System.out.println("done");
	}

}
