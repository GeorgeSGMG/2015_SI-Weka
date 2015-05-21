package es.weka;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class Temp2 {

	public static void muestraContenido(String archivo) throws FileNotFoundException, IOException {
        String cadena;
        FileReader f = new FileReader(archivo);
        BufferedReader b = new BufferedReader(f);
        FileWriter fichero = null;
        PrintWriter pw = null;
        try
        {
            fichero = new FileWriter("src/example/resources/masocada2.txt");
            pw = new PrintWriter(fichero);
            while((cadena = b.readLine())!=null) {
                String s = cadena;
                if (s.charAt(1)=='0') {
                	s = "{" + s.substring(7, s.length()-1) + ",1491 ham}";
                }
            	pw.println(s);
            }
            b.close();
        } catch (Exception e) {}
    }
 
    public static void main(String[] args) throws IOException {
        muestraContenido("src/example/resources/masocada.txt");
    }

}
