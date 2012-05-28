package pl.pw.edu.prir.tsole;

public class Manual {

	public static void parseArgs(String[] args) {
		
	}
	
	public static void printHelp() {
		System.out.println("-h					- pomoc\n" +
							"-a <ścieżka>		- ścieżka do pliku z równaniami macierzy A\n" +
							"-b <ścieżka>		- ścieżka do pliku z równaniami macierzy B\n" +
							"-m <nazwa>			- nazwa metody rozwiązywania układów równań:\n" +
							"					- gauss			- metoda eliminacji Gaussa\n" +
							"					- gauss-jordan	- metoda eliminacju Gaussa-Jordana\n" +
							"					- crout			- metoda Crouta\n" +
							"					- all			- wszystkie zaimplementowane metody\n" +
							"-o <ścieżka>		- ścieżka do pliku z wynikami\n");
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
