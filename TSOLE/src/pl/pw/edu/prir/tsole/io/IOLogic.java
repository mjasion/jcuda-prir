package pl.pw.edu.prir.tsole.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;

import org.apache.log4j.Logger;

public class IOLogic {
	private static Logger log = Logger.getLogger(IOLogic.class);
	public static float[][] readMatrix(String path) {
		return readMatrix(new File(path));
	}

	public static float[][] readMatrix(File file) {
		float[][] matrix = null;
		if(file == null)
			return null;
		
		FileReader fr;
		BufferedReader in;
		try {
			fr = new FileReader(file);
			in = new BufferedReader(fr);
			String line;
			String[] tabLine;
			int m = -1; // wiersze
			int n = -1; // kolumny
			
			line = in.readLine();
			while(line.startsWith("#"))
				line = in.readLine();
			
			tabLine = line.split("x");
			m = Integer.parseInt(tabLine[0].trim());
			n = Integer.parseInt(tabLine[1].trim());
			
			matrix = new float[m][n];
			int rowIndex = 0;
			while ((line = in.readLine()) != null) {
				if (line.startsWith("#") || line.equals("")) // komentarz - pusta linia
					continue;
				
				tabLine = line.split(" ");
				if(tabLine.length > n) {
					log.error("BLAD PARSOWANIA LINII " + (rowIndex+1));
					log.error("POMIJAM LINIE " + (rowIndex+1));
					continue;
				}
				for(int i=0; i<n; i++) {
					try {
						matrix[rowIndex][i] = Float.parseFloat(tabLine[i]);
					} catch (NumberFormatException ex) {
						log.error("Niepoprawny format liczby: " + tabLine[i]);
						log.warn("Zastepcza wartosc: " + rowIndex + "x" + i + " = 0.0");
					}
				}
				rowIndex++;
			}
			if(rowIndex < m) {
				log.info("Zmienszam tablice z " + m + " do " + rowIndex + " wierszy");
				matrix = zmmniejszTablice(matrix, rowIndex, n);
				log.info("Nowy rozmiar tablicy " + matrix.length + "x" + matrix[0].length);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return matrix;
	}

	private static float[][] zmmniejszTablice(float[][] matrix, int rowIndex, int n) {
		float[][] m = new float[rowIndex][n];
		for(int i=0; i<rowIndex; i++){
			m[i] = Arrays.copyOf(matrix[i], n);
		}
		
		return m;
		
	}
}
