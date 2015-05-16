package es.weka;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import es.weka.Decider;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import es.weka.InstanceBuilder.BuildResponse;

public class MiniExample {

        private enum TrecFeatures {aeiou, a, account, all, ams, and, anymore, app, are, as, ave, award, awarded, been, bonus, boston, buy, call, caller, callers, callertune, can, car, changed, claim, code, contact, copy, cuz, didn, don, draw, driving, ending, expires, flights, friends, from, gave, gent, gone, guaranteed, guess, has, have, holiday, i, identifier, inc, is, it, its, k52, landline, last, legal, live, location, looking, m, match, me, mean, melle, men, min, minnaminunginte, mobile, my, name, no, not, nothing, numbers, nurungu, nyc, okay, on, once, only, operator, or, oru, our, page, per, pick, please, points, press, price, prize, quoting, rates, re, receive, request, right, say, says, search, selected, set, shows, signin, something, speak, specially, standard, statement, still, t, test, text, that, them, to, todays, trying, u, unredeemed, up, ur, urgent, valid, vettam, voda, was, we, weekends, wen, winner, won, wow, x, xx, y, yet, you, your};
        private enum TrecClass {spam,ham} ;

        @SuppressWarnings({ "unchecked", "rawtypes" })
		public static void main(String[] args) throws Exception {

                //build a decider, which knows:
                // - what attributes are involved
                // - what data types these attributes are (in this case all numeric)
                // - what the expected output is (in this case, an enum, but numeric and binary is also doable)
                Decider<TrecFeatures,TrecClass> decider = 
                        new DeciderBuilder("TrecDecider", TrecFeatures.class)
                .setDefaultAttributeTypeNumeric()
                .setClassAttributeTypeEnum("$class$", TrecClass.class)
                .build();

                //load training data from file
                //this will check that attributes match TrecFeatures enum, that class attribute is named "class" and is of correct type, and so on.
                Dataset<TrecFeatures,TrecClass> dataset = decider.createNewDataset() ;
                dataset.load(new File("src/example/resources/result.arff")) ;

                //train a classifier using loaded training data.
                decider.train(new SMO(), dataset) ;

                //save the classifier so we could skip training in future
                //unfortunately this doesn't make any checks to see if classifier was trained on expected attributes
                //any idea how one would do that?
                decider.save(new File("src/example/resources/trec.model")) ;

                //load the classifier saved previously
                //unfortunately this doesn't make any checks to see if classifier was trained on expected attributes
                decider.load(new File("src/example/resources/trec.model")) ;

                String correo = "You are a winner U have been specially selected 2 receive £1000";
                System.out.println(correo);
        		correo = " " + correo.toLowerCase().replace("\n", " ").replace("\t", " ").replace(" \"", " ").replace("\" ", " ").replace(" ", "  ").replace(". ", " ").replace(", ", " ").replace(": ", " ").replace("; ", " ").replace(" ¿", " ").replace("? ", " ").replace(" ¡", " ").replace("! ", " ") + " ";
                System.out.println(correo);
        		
                //build an instance that we can classify
                //this will check that all attributes are set (optional) and that values are the correct type.
                Instance i = decider.getInstanceBuilder()
                .setAttributeMissingResponse(BuildResponse.THROW_ERROR)
                .setAttribute(TrecFeatures.aeiou, 0)
                .setAttribute(TrecFeatures.a, contarPalabras(correo, " a "))
                .setAttribute(TrecFeatures.account, contarPalabras(correo, " account "))
                .setAttribute(TrecFeatures.all, contarPalabras(correo, " all "))
                .setAttribute(TrecFeatures.ams, contarPalabras(correo, " ams "))
                .setAttribute(TrecFeatures.and, contarPalabras(correo, " and "))
                .setAttribute(TrecFeatures.anymore, contarPalabras(correo, " anymore "))
                .setAttribute(TrecFeatures.app, contarPalabras(correo, " app "))
                .setAttribute(TrecFeatures.are, contarPalabras(correo, " are "))
                .setAttribute(TrecFeatures.as, contarPalabras(correo, " as "))
                .setAttribute(TrecFeatures.ave, contarPalabras(correo, " ave "))
                .setAttribute(TrecFeatures.award, contarPalabras(correo, " award "))
                .setAttribute(TrecFeatures.awarded, contarPalabras(correo, " awarded "))
                .setAttribute(TrecFeatures.been, contarPalabras(correo, " been "))
                .setAttribute(TrecFeatures.bonus, contarPalabras(correo, " bonus "))
                .setAttribute(TrecFeatures.boston, contarPalabras(correo, " boston "))
                .setAttribute(TrecFeatures.buy, contarPalabras(correo, " buy "))
                .setAttribute(TrecFeatures.call, contarPalabras(correo, " call "))
                .setAttribute(TrecFeatures.caller, contarPalabras(correo, " caller "))
                .setAttribute(TrecFeatures.callers, contarPalabras(correo, " callers "))
                .setAttribute(TrecFeatures.callertune, contarPalabras(correo, " callertune "))
                .setAttribute(TrecFeatures.can, contarPalabras(correo, " can "))
                .setAttribute(TrecFeatures.car, contarPalabras(correo, " car "))
                .setAttribute(TrecFeatures.changed, contarPalabras(correo, " changed "))
                .setAttribute(TrecFeatures.claim, contarPalabras(correo, " claim "))
                .setAttribute(TrecFeatures.code, contarPalabras(correo, " code "))
                .setAttribute(TrecFeatures.contact, contarPalabras(correo, " contact "))
                .setAttribute(TrecFeatures.copy, contarPalabras(correo, " copy "))
                .setAttribute(TrecFeatures.cuz, contarPalabras(correo, " cuz "))
                .setAttribute(TrecFeatures.didn, contarPalabras(correo, " didn "))
                .setAttribute(TrecFeatures.don, contarPalabras(correo, " don "))
                .setAttribute(TrecFeatures.draw, contarPalabras(correo, " draw "))
                .setAttribute(TrecFeatures.driving, contarPalabras(correo, " driving "))
                .setAttribute(TrecFeatures.ending, contarPalabras(correo, " ending "))
                .setAttribute(TrecFeatures.expires, contarPalabras(correo, " expires "))
                .setAttribute(TrecFeatures.flights, contarPalabras(correo, " flights "))
                .setAttribute(TrecFeatures.friends, contarPalabras(correo, " friends "))
                .setAttribute(TrecFeatures.from, contarPalabras(correo, " from "))
                .setAttribute(TrecFeatures.gave, contarPalabras(correo, " gave "))
                .setAttribute(TrecFeatures.gent, contarPalabras(correo, " gent "))
                .setAttribute(TrecFeatures.gone, contarPalabras(correo, " gone "))
                .setAttribute(TrecFeatures.guaranteed, contarPalabras(correo, " guaranteed "))
                .setAttribute(TrecFeatures.guess, contarPalabras(correo, " guess "))
                .setAttribute(TrecFeatures.has, contarPalabras(correo, " has "))
                .setAttribute(TrecFeatures.have, contarPalabras(correo, " have "))
                .setAttribute(TrecFeatures.holiday, contarPalabras(correo, " holiday "))
                .setAttribute(TrecFeatures.i, contarPalabras(correo, " i "))
                .setAttribute(TrecFeatures.identifier, contarPalabras(correo, " identifier "))
                .setAttribute(TrecFeatures.inc, contarPalabras(correo, " inc "))
                .setAttribute(TrecFeatures.is, contarPalabras(correo, " is "))
                .setAttribute(TrecFeatures.it, contarPalabras(correo, " it "))
                .setAttribute(TrecFeatures.its, contarPalabras(correo, " its "))
                .setAttribute(TrecFeatures.k52, contarPalabras(correo, " k52 "))
                .setAttribute(TrecFeatures.landline, contarPalabras(correo, " landline "))
                .setAttribute(TrecFeatures.last, contarPalabras(correo, " last "))
                .setAttribute(TrecFeatures.legal, contarPalabras(correo, " legal "))
                .setAttribute(TrecFeatures.live, contarPalabras(correo, " live "))
                .setAttribute(TrecFeatures.location, contarPalabras(correo, " location "))
                .setAttribute(TrecFeatures.looking, contarPalabras(correo, " looking "))
                .setAttribute(TrecFeatures.m, contarPalabras(correo, " m "))
                .setAttribute(TrecFeatures.match, contarPalabras(correo, " match "))
                .setAttribute(TrecFeatures.me, contarPalabras(correo, " me "))
                .setAttribute(TrecFeatures.mean, contarPalabras(correo, " mean "))
                .setAttribute(TrecFeatures.melle, contarPalabras(correo, " melle "))
                .setAttribute(TrecFeatures.men, contarPalabras(correo, " men "))
                .setAttribute(TrecFeatures.min, contarPalabras(correo, " min "))
                .setAttribute(TrecFeatures.minnaminunginte, contarPalabras(correo, " minnaminunginte "))
                .setAttribute(TrecFeatures.mobile, contarPalabras(correo, " mobile "))
                .setAttribute(TrecFeatures.my, contarPalabras(correo, " my "))
                .setAttribute(TrecFeatures.name, contarPalabras(correo, " name "))
                .setAttribute(TrecFeatures.no, contarPalabras(correo, " no "))
                .setAttribute(TrecFeatures.not, contarPalabras(correo, " not "))
                .setAttribute(TrecFeatures.nothing, contarPalabras(correo, " nothing "))
                .setAttribute(TrecFeatures.numbers, contarPalabras(correo, " numbers "))
                .setAttribute(TrecFeatures.nurungu, contarPalabras(correo, " nurungu "))
                .setAttribute(TrecFeatures.nyc, contarPalabras(correo, " nyc "))
                .setAttribute(TrecFeatures.okay, contarPalabras(correo, " okay "))
                .setAttribute(TrecFeatures.on, contarPalabras(correo, " on "))
                .setAttribute(TrecFeatures.once, contarPalabras(correo, " once "))
                .setAttribute(TrecFeatures.only, contarPalabras(correo, " only "))
                .setAttribute(TrecFeatures.operator, contarPalabras(correo, " operator "))
                .setAttribute(TrecFeatures.or, contarPalabras(correo, " or "))
                .setAttribute(TrecFeatures.oru, contarPalabras(correo, " oru "))
                .setAttribute(TrecFeatures.our, contarPalabras(correo, " our "))
                .setAttribute(TrecFeatures.page, contarPalabras(correo, " page "))
                .setAttribute(TrecFeatures.per, contarPalabras(correo, " per "))
                .setAttribute(TrecFeatures.pick, contarPalabras(correo, " pick "))
                .setAttribute(TrecFeatures.please, contarPalabras(correo, " please "))
                .setAttribute(TrecFeatures.points, contarPalabras(correo, " points "))
                .setAttribute(TrecFeatures.press, contarPalabras(correo, " press "))
                .setAttribute(TrecFeatures.price, contarPalabras(correo, " price "))
                .setAttribute(TrecFeatures.prize, contarPalabras(correo, " prize "))
                .setAttribute(TrecFeatures.quoting, contarPalabras(correo, " quoting "))
                .setAttribute(TrecFeatures.rates, contarPalabras(correo, " rates "))
                .setAttribute(TrecFeatures.re, contarPalabras(correo, " re "))
                .setAttribute(TrecFeatures.receive, contarPalabras(correo, " receive "))
                .setAttribute(TrecFeatures.request, contarPalabras(correo, " request "))
                .setAttribute(TrecFeatures.right, contarPalabras(correo, " right "))
                .setAttribute(TrecFeatures.say, contarPalabras(correo, " say "))
                .setAttribute(TrecFeatures.says, contarPalabras(correo, " says "))
                .setAttribute(TrecFeatures.search, contarPalabras(correo, " search "))
                .setAttribute(TrecFeatures.selected, contarPalabras(correo, " selected "))
                .setAttribute(TrecFeatures.set, contarPalabras(correo, " set "))
                .setAttribute(TrecFeatures.shows, contarPalabras(correo, " shows "))
                .setAttribute(TrecFeatures.signin, contarPalabras(correo, " signin "))
                .setAttribute(TrecFeatures.something, contarPalabras(correo, " something "))
                .setAttribute(TrecFeatures.speak, contarPalabras(correo, " speak "))
                .setAttribute(TrecFeatures.specially, contarPalabras(correo, " specially "))
                .setAttribute(TrecFeatures.standard, contarPalabras(correo, " standard "))
                .setAttribute(TrecFeatures.statement, contarPalabras(correo, " statement "))
                .setAttribute(TrecFeatures.still, contarPalabras(correo, " still "))
                .setAttribute(TrecFeatures.t, contarPalabras(correo, " t "))
                .setAttribute(TrecFeatures.test, contarPalabras(correo, " test "))
                .setAttribute(TrecFeatures.text, contarPalabras(correo, " text "))
                .setAttribute(TrecFeatures.that, contarPalabras(correo, " that "))
                .setAttribute(TrecFeatures.them, contarPalabras(correo, " them "))
                .setAttribute(TrecFeatures.to, contarPalabras(correo, " to "))
                .setAttribute(TrecFeatures.todays, contarPalabras(correo, " todays "))
                .setAttribute(TrecFeatures.trying, contarPalabras(correo, " trying "))
                .setAttribute(TrecFeatures.u, contarPalabras(correo, " u "))
                .setAttribute(TrecFeatures.unredeemed, contarPalabras(correo, " unredeemed "))
                .setAttribute(TrecFeatures.up, contarPalabras(correo, " up "))
                .setAttribute(TrecFeatures.ur, contarPalabras(correo, " ur "))
                .setAttribute(TrecFeatures.urgent, contarPalabras(correo, " urgent "))
                .setAttribute(TrecFeatures.valid, contarPalabras(correo, " valid "))
                .setAttribute(TrecFeatures.vettam, contarPalabras(correo, " vettam "))
                .setAttribute(TrecFeatures.voda, contarPalabras(correo, " voda "))
                .setAttribute(TrecFeatures.was, contarPalabras(correo, " was "))
                .setAttribute(TrecFeatures.we, contarPalabras(correo, " we "))
                .setAttribute(TrecFeatures.weekends, contarPalabras(correo, " weekends "))
                .setAttribute(TrecFeatures.wen, contarPalabras(correo, " wen "))
                .setAttribute(TrecFeatures.winner, contarPalabras(correo, " winner "))
                .setAttribute(TrecFeatures.won, contarPalabras(correo, " won "))
                .setAttribute(TrecFeatures.wow, contarPalabras(correo, " wow "))
                .setAttribute(TrecFeatures.x, contarPalabras(correo, " x "))
                .setAttribute(TrecFeatures.xx, contarPalabras(correo, " xx "))
                .setAttribute(TrecFeatures.y, contarPalabras(correo, " y "))
                .setAttribute(TrecFeatures.yet, contarPalabras(correo, " yet "))
                .setAttribute(TrecFeatures.you, contarPalabras(correo, " you "))
                .setAttribute(TrecFeatures.your, contarPalabras(correo, " your "))
                .build() ;

                //note: building training instances algorithmically is done in much the same way

                //now identify the class of the instance we built
                TrecClass c = decider.getDecision(i);
                System.out.println(c) ;

                //and get some details about how this decision was made 
                HashMap<TrecClass, Double> distributions = decider.getDecisionDistribution(i) ;
                for (Map.Entry<TrecClass, Double> e:distributions.entrySet()) 
                        System.out.println(e.getKey() + ": " + e.getValue()) ;            
        }
        
        public static int contarPalabras(String sTexto, String sTextoBuscado) {
    		int contador = 0;
    		while (sTexto.indexOf(sTextoBuscado) > -1) {
    		      sTexto = sTexto.substring(sTexto.indexOf(
    		        sTextoBuscado)+sTextoBuscado.length(),sTexto.length());
    		      contador++; 
    		}
    		return contador;
    	}
}