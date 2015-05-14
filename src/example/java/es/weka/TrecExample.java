package es.weka;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import es.weka.Decider;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import es.weka.InstanceBuilder.BuildResponse;

public class TrecExample {

        private enum TrecFeatures {contents};
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
                dataset.load(new File("src/example/resources/smsspam.small.test.arff")) ;

                //train a classifier using loaded training data.
                decider.train(new SMO(), dataset) ;

                //save the classifier so we could skip training in future
                //unfortunately this doesn't make any checks to see if classifier was trained on expected attributes
                //any idea how one would do that?
                decider.save(new File("src/example/resources/trec.model")) ;

                //load the classifier saved previously
                //unfortunately this doesn't make any checks to see if classifier was trained on expected attributes
                decider.load(new File("src/example/resources/trec.model")) ;

                //build an instance that we can classify
                //this will check that all attributes are set (optional) and that values are the correct type.
                Instance i = decider.getInstanceBuilder()
                .setAttributeMissingResponse(BuildResponse.THROW_ERROR)
                .setAttribute(TrecFeatures.contents, " murphy.debian.org (murphy.debian.org  \n \tby speedy.uwaterloo.ca  with esmtp id  \n \tfor  sun,  apr    \n received: from localhost (localhost  \n \tby murphy.debian.org (postfix) with qmqp\n \n \tid  sun,   apr    (cdt)\n \n old-return-path:  \n x-spam-checker-version: spamassassin   on murphy.debian.org\n \n x-spam-level: \n \n x-spam-status: no,     \n \n  \n x-original-to:  \n received: from xenon.savoirfairelinux.net (savoirfairelinux.net  \n \tby murphy.debian.org (postfix) with esmtp id  \n \tfor  sun,   apr    (cdt)\n \n received: from    \n \tby xenon.savoirfairelinux.net (postfix) with esmtp id  \n \tfor  sun,   apr    (edt)\n \n message-id:  \n date: sun,  apr    \n from: yan morin  \n user-agent: icedove   \n mime-version:  \n to:  \n subject: typo in  \n x-enigmail-version:  \n content-type:   \n content-transfer-encoding:  \n x-rc-spam:  \n x-rc-virus:  \n x-rc-spam:  \n resent-message-id:  \n resent-from:  \n x-mailing-list:  \n \n x-loop:  \n list-id:  \n list-post:  \n list-help:  \n list-subscribe:  \n list-unsubscribe:  \n precedence: list\n \n resent-sender:  \n resent-date: sun,   apr    (cdt)\n \n status: ro\n \n content-length:  \n lines:  \n \n \n hi, i\'ve just updated from the gulus and i check on other mirrors.\n \n it seems there is a little typo in  file\n \n \n \n example:\n \n  \n  \n \n \n \"testing, or lenny.  access this release through   the\n \n current tested development snapshot is named etch.  packages which\n \n have been tested in unstable and passed automated tests propogate to\n \n this release.\"\n \n \n \n etch should be replace by lenny like in the readme.html\n \n \n \n \n \n \n \n -- \n \n yan morin\n \n consultant en logiciel libre\n \n  \n  \n \n \n \n \n -- \n \n to unsubscribe, email to  \n with a subject of \"unsubscribe\". trouble? contact  \n \n \n ".toString())
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
}