package pl.pw.edu.prir.tsole;

public class Manual {

	public static String METHOD_ALL = "all";
	public static String METHOD_GAUSS = "gasuss";
	public static String METHOD_GAUSS_JORDAN = "gauss-jordan";
	
	public static int parseArgs(String[] args) {
		if (args.length == 1 && args[0].equals("-h")) {
			printHelp();
			return 1;
		}
		else if (args.length == 6 && args[0].equals("-a") && args[2].equals("-b") && args[4].equals("-m")) {
			Main.setPathToMatrixA(args[1]);
			Main.setPathToMatrixB(args[3]);
			Main.setMethodSelected(args[5]);
			return 0;
		}
		else if (args.length == 8 && args[0].equals("-a") && args[2].equals("-b") && args[4].equals("-m") && args[6].equals("-o") && 
				(args[5].equals(METHOD_ALL) || args[5].equals(METHOD_GAUSS) || args[5].equals(METHOD_GAUSS_JORDAN))) {
			Main.setPathToMatrixA(args[1]);
			Main.setPathToMatrixB(args[3]);
			Main.setMethodSelected(args[5]);
			Main.setPathToOutputFile(args[7]);
			return 0;
		}
		
		if (args[args.length-1].equals("-v"))
			Main.setVerbose(true);
		
		return 1;
	}
	
	public static void printHelp() {
		System.out.println("-h					- pomoc\n" +
							"-a <ścieżka>		- ścieżka do pliku z równaniami macierzy A\n" +
							"-b <ścieżka>		- ścieżka do pliku z równaniami macierzy B\n" +
							"-m <nazwa>			- nazwa metody rozwiązywania układów równań:\n" +
							"					- gauss			- metoda eliminacji Gaussa\n" +
							"					- gauss-jordan	- metoda eliminacju Gaussa-Jordana\n" +
							"					- all			- wszystkie zaimplementowane metody\n" +
							"-o <ścieżka>		- ścieżka do pliku z wynikami\n" +
							"-v					- wyświetla wszystkie logi CUDA\n");
	}

	public static void printWelcome() {
		System.out.println("M MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMMM.  .. :MM .~MMMI. MM     ..+MMMMM  .MMMMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMM   MMM. MM  ~MMMI  MM   MMM. ZMMMM.  IMMMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMM  NMMMMMMM  ~MMMI  MM   MMM= .MMM  D  MMMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMM  NMMMMMMM  ~MMMI  MM   MMM+  MMI  M~  MMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMM  8MMMMMMM  ~MMMI  MM   MMM. .MM. Z$Z  MMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMM   MMM .MMD. MMM   MM  .MMM  NM7 ..  .  MMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMMM= .  .MMMMM    .IMMM      .MMM .MMMMM. MI,MMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM M\n"+
							"M MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM M\n"+
							"		Programowanie równoległe i rozproszone\n"+
							"	Marcin Jasion   Marek Sarnecki   Tobiasz Siemiński\n\n\n");
	}
}
