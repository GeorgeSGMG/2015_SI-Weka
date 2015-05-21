package es.weka;

public class Temp1 {
	public static void main(String[] args) throws Exception {
		String correo = "a account all all; \"ams\" all ams \tall ams, as\n ave";
		correo = " " + correo.replace("\n", " ").replace("\t", " ").replace(" \"", " ").replace("\" ", " ").replace(" ", "  ").replace(". ", " ").replace(", ", " ").replace(": ", " ").replace("; ", " ").replace(" ¿", " ").replace("? ", " ").replace(" ¡", " ").replace("! ", " ") + " ";
		String palabras = "abhorred, able, about, above, absolutely, access, account, achieve, acrobat, action, actually, add, address, adobe, advantage, after, again, age, aggresive, ago, agree, al, albert, alert, alice, all, allows, almost, alone, along, already, also, although, always, am, amazing, american, among, amount, an, anatrim, and, andrea, annoyed, another, antichrist, any, anyone, anything, anyway, anywhere, appetite, approaches, approved, april, are, area, arial, arms, around, art, article, articles, as, asatru, ask, asked, assume, at, ate, author, availability, available, away, b, back, bad, balance, bank, barnwell, based, bayonne, be, became, because, become, bed, been, before, began, behind, being, believe, below, benefit, best, better, between, beverage, big, bill, bird, bit, black, blend, blue, body, book, both, boy, boys, br, brand, brocker, broker, brought, bullish, business, but, buy, by, c, california, call, called, came, can, canadian, cannot, capital, care, casino, caught, cause, caused, cdyv, celtic, center, cents, century, chance, change, check, child, childhood, children, choice, choose, chose, christ, christian, christianity, christians, cialis, circle, city, click, climate, coffee, cold, com, come, comes, coming, comments, common, companies, company, compensation, competing, complete, computer, con, contact, continued, corporation, could, count, country, course, coyote, creative, culture, current, customer, customers, d, dangerous, dark, data, day, days, de, dead, deal, dear, debt, deceived, decided, deep, definitions, degree, del, delivery, depression, design, destroy, detested, did, difference, different, difficult, digital, directly, discussion, doctor, doctrines, does, doesna, doing, dona, done, dose, down, download, dr, drained, drive, drug, drugs, druidism, during, dysfunction, each, early, earth, east, eat, eating, ed, eden, effect, effective, effects, either, el, electronic, elegant, em, email, en, end, energy, enlargement, ennis, enough, enterprise, erectile, erection, erections, error, es, established, evanelists, evangelical, even, ever, every, everyone, everything, excellent, exercises, existed, expected, experience, exquisite, extra, eyes, face, fact, far, fast, fat, feel, feeling, feet, felt, few, field, fight, figure, filtered, find, fine, fire, first, fit, five, fiz, flesh, floor, following, food, forget, form, format, formula, fotografía, found, four, free, friday, friend, friends, from, full, further, future, g, gain, game, games, gasses, generic, get, getting, gift, girl, give, giving, go, god, goes, going, gone, good, got, government, granted, great, greenhouse, greenwood, group, groups, growing, growth, gte, had, half, hand, hands, hans, happiest, happy, hard, hardware, has, hate, have, he, head, headline, health, hear, heard, hellinic, help, helps, helvetica, her, herbal";
		String[] palabrasArray = palabras.split(", ");
		for (int i = 0; i < palabrasArray.length; i++) {
			System.out.println(".setAttribute(TrecFeatures." + palabrasArray[i] + ", contarPalabras(correo, \" " + palabrasArray[i] + " \"))");
		}
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
