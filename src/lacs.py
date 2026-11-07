# lacs suisses (swissBATHY3D), bodensee/lacleman exclus -- MNT suisse hors frontiere
LAKES_CH = [
    "lacdejoux", "lungernsee", "aegerisee", "baldeggersee", "hallwilersee",
    "bielersee", "brienzersee", "lacneuchatel", "lagomaggiore",
]

# lacs francais : nom du dossier sur
# https://adour-garonne.eaufrance.fr/upload/DATA/SIG/BATHYMETRIE/xyz_pointcloud/
# (liste relevee le 06/11/2026). L8 n'existe pas sur le serveur.
# Exclus le 03/08 (donnees source corrompues) : L5, L14, L20, L21, L24, L25, L75, L92.
# Attention : L87 a deux releves (L87_2011 et L87_AQUABIO_20130831), et les deux
# donnent le meme LAKE_ID "L87". Je ne sais plus lequel servait avant le 06/11 ; depuis,
# c'est L87_2011 (extraction des 107 lacs et test final du 07/11).
LAKES_FR_FOLDERS = {
    "L1": "L1_2011", "L2": "L2_2011", "L3": "L3_2010", "L4": "L4_AQUABIO_20130301",
    "L6": "L6_2011", "L7": "L7_2010", "L9": "L9_2011", "L10": "L10_2010",
    "L11": "L11_2011", "L12": "L12_ASCONIT_2014", "L13": "L13_2010", "L15": "L15_2010",
    "L16": "L16_ECOGEA_20140716", "L17": "L17_AQUABIO_20120827", "L18": "L18_AQUABIO_20130613",
    "L19": "L19_2010", "L22": "L22_ASCONIT_2014", "L23": "L23_2010",
    "L26": "L26_2010", "L27": "L27_AQUABIO_20140331", "L28": "L28_AQUABIO_20130325",
    "L29": "L29_AQUABIO_20120613", "L30": "L30_AQUABIO_20140418", "L31": "L31_2010",
    "L32": "L32_AQUABIO_20130402", "L33": "L33_AQUABIO_20130228", "L34": "L34_AQUABIO_20120622",
    "L35": "L35_2010", "L36": "L36_2010", "L37": "L37_ASCONIT_2013", "L38": "L38_2010",
    "L39": "L39_2010", "L40": "L40_2011", "L41": "L41_ASCONIT_2014", "L42": "L42_2010",
    "L43": "L43_ECOGEA_20140722", "L44": "L44_AQUABIO_20120621", "L45": "L45_2010",
    "L46": "L46_AQUABIO_20130830", "L47": "L47_2010", "L48": "L48_2011",
    "L49": "L49_AQUABIO_20120828", "L50": "L50_AQUABIO_20130327", "L51": "L51_2010",
    "L52": "L52_2010", "L53": "L53_2010", "L54": "L54_2010", "L55": "L55_2010",
    "L56": "L56_2010", "L57": "L57_2010", "L58": "L58_2010", "L59": "L59_ASCONIT_2013",
    "L60": "L60_AQUABIO_20130328", "L61": "L61_2010", "L62": "L62_2011",
    "L63": "L63_AQUABIO_20130328", "L64": "L64_AQUABIO_20140418", "L65": "L65_2010",
    "L66": "L66_ECOGEA_20131011", "L67": "L67_2010", "L68": "L68_ASCONIT_2013",
    "L69": "L69_ASCONIT_2013", "L70": "L70_2010", "L71": "L71_2010", "L72": "L72_2010",
    "L73": "L73_ECOGEA_20140717", "L74": "L74_AQUABIO_20120619", "L76": "L76_2008",
    "L77": "L77_AQUABIO_20120618", "L78": "L78_AQUABIO_20120621", "L79": "L79_2010",
    "L80": "L80_ASCONIT_2012", "L81": "L81_AQUABIO_20130329", "L82": "L82_AQUABIO_20130329",
    "L83": "L83_ASCONIT_2014", "L84": "L84_2011", "L85": "L85_ASCONIT_2014", "L86": "L86_2010",
    "L87": "L87_2011", "L88": "L88_ECOGEA_20130801", "L89": "L89_2010", "L90": "L90_2010",
    "L91": "L91_2011", "L93": "L93_ASCONIT_2014", "L94": "L94_2010", "L95": "L95_2010",
    "L96": "L96_2011", "L97": "L97_2011", "L98": "L98_2010", "L99": "L99_AQUABIO_20140418",
    "L100": "L100_AQUABIO_20140519", "L101": "L101_2010", "L102": "L102_2010",
    "L103": "L103_2010", "L104": "L104_AQUABIO_20120621", "L105": "L105_2010",
    "L106": "L106_2010", "L107": "L107_AQUABIO_20130227",
}
LAKES_FR = list(LAKES_FR_FOLDERS)

# lacs qui ont deja servi a prendre des decisions (jours 4 a 20 : cible en ratio,
# coefficient isoperimetrique, masque de Sobel...). On ne peut plus s'en servir
# pour le test final, sinon le score est trop optimiste.
LAKES_ALREADY_SEEN = LAKES_CH + ["L1", "L2", "L9", "L17", "L30", "L40", "L50", "L60", "L66", "L70", "L80", "L90"]
